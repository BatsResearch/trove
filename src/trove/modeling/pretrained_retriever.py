"""Abstraction on top of dense encoders for information retrieval."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.distributed as dist
from torch import nn
from transformers import TrainingArguments
from transformers.file_utils import ModelOutput

from . import modeling_utils
from .loss_base import RetrievalLoss
from .model_args import ModelArguments
from .pretrained_encoder import PretrainedEncoder


@dataclass
class RetrieverOutput(ModelOutput):
    """Contains the output of the retriever model."""

    query: Optional[torch.Tensor] = None
    """Query embeddings"""
    passage: Optional[torch.Tensor] = None
    """Passage embeddings"""
    loss: Optional[torch.Tensor] = None
    """Calculated loss"""
    logits: Optional[torch.Tensor] = None
    """similarity score between queries and passages"""


class PretrainedRetriever(nn.Module):
    def __init__(
        self,
        model_args: ModelArguments,
        encoder: nn.Module,
        preprocess_only: bool = False,
        format_query: Optional[Callable] = None,
        format_passage: Optional[Callable] = None,
        append_eos_token: Optional[bool] = None,
        loss_extra_kwargs: Optional[Dict] = None,
    ) -> None:
        """A base class for training/inference with different retrievers.

        Args:
            model_args (ModelArguments): config specifying the model and loss to use.
                Currently, we only use ``model_args`` to instantiate the loss (it might also be saved by some loggers).
                To instantiate both model and loss, use :meth:`from_model_args`.
            encoder (nn.Module): encoder model to use. It must expose ``encode_query()`` and ``encode_passage()`` methods.
                ``encoder`` is also expected to provide a ``save_pretrained()`` method but only if you call :meth:`save_pretrained`
            preprocess_only (bool): if true, do not instantiate loss module.
                You should also pass this to :meth:`from_model_args` if you
                do not want to load the model parameters. See ``PretrainedEncoder.__init__`` for details.
            format_query (Optional[Callable]): Callable similar to `PretrainedEncoder.format_query`.
                If provided, it is prioritized over ``encoder.format_query()``.
                It is not used by this class internally. It is just exposed as a convenience method to
                keep everything needed to encode a query in one place.
            format_passage (Optional[Callable]): Callable similar to ``PretrainedEncoder.format_passage()``
                If provided, it is prioritized over ``encoder.format_passage``.
                It is not used by this class internally. It is just exposed as a convenience method to
                keep everything needed to encode a query in one place.
            append_eos_token (Optional[bool]): Similar to ``PretrainedEncoder.append_eos_token``
                If provided, it is prioritized over ``encoder.append_eos_token``.
                It is not used by this class internally. It is just exposed as a convenience method to
                keep everything needed to encode a query in one place.
            loss_extra_kwargs (Optional[Dict]): If given, these are passed to ``RetrievalLoss.__init__()`` as keyword arguments.
        """
        super().__init__()

        self.args = model_args
        self.encoder = encoder

        # The encoder model should take care of these
        # Expose these to remain compatible with a subset (and *NOT* all) of tools
        # (e.g., transformers.Trainer) that expect a transformers.PretrainedModel instance
        modeling_utils.add_model_apis_to_wrapper(wrapper=self, model=encoder)

        if not preprocess_only:
            _model_param = next(iter(self.encoder.parameters()))
            _loss_kwargs = dict(dtype=_model_param.dtype, device=_model_param.device)
            if loss_extra_kwargs is not None:
                _loss_kwargs = {**_loss_kwargs, **loss_extra_kwargs}
            self.loss: RetrievalLoss = RetrievalLoss.from_model_args(
                args=self.args, **_loss_kwargs
            )
        else:
            self.loss = None

        if format_query is not None:
            self.format_query = format_query
        else:
            self.format_query = getattr(encoder, "format_query", None)

        if format_passage is not None:
            self.format_passage = format_passage
        else:
            self.format_passage = getattr(encoder, "format_passage", None)

        if append_eos_token is not None:
            self.append_eos_token = append_eos_token
        else:
            self.append_eos_token = getattr(encoder, "append_eos_token", None)

        if hasattr(encoder, "similarity_fn"):
            self.similarity_fn = encoder.similarity_fn
        else:
            self.similarity_fn = lambda x, y: x @ y.T

        self.process_rank = None
        self.world_size = None
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    @classmethod
    def from_model_args(
        cls,
        args: ModelArguments,
        model_name_or_path: Optional[str] = None,
        training_args: Optional[TrainingArguments] = None,
        loss_extra_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        """Instantiate the retriever according based on the given args.

        Args:
            args (ModelArguments): config used to instantiate the encoder and loss modules.
            model_name_or_path (Optional[str]): name of the encoder model to load.
                If not provided, use ``args.model_name_or_path``. You should almost never
                need to use this.
            training_args (TrainingArguments): passed to ``PretrainedEncoder.from_model_args()``.
                It is used to enable gradient checkpointing if needed.
            loss_extra_kwargs (Optional[Dict]): passed to ``BiEncoderRetriever.__init__()`` which then passes them to ``RetrievalLoss.__init__()``.
            **kwargs: extra keyword arguments passed to ``BiEncoderRetriever.__init__()`` and
                ``PretrainedEncoder.from_model_args()``. If you want to avoid loading model parameters
                and only load methods and attributes required for pre-processing, you should pass
                ``preprocess_only=True`` as part of this kwargs. See ``BiEncoderRetriever.__init__()`` and ``PretrainedEncoder.__init__()``
                for details.
        Returns:
            an instance of one of `PretrainedRetriever` subclasses.
        """
        if args.encoder_class is None or args.encoder_class.lower() == "none":
            m_name = (
                args.model_name_or_path
                if model_name_or_path is None
                else model_name_or_path
            )
            trove_conf = modeling_utils.load_trove_retriever_config(
                model_name=m_name, model_revision=args.model_revision
            )
            if trove_conf is not None and "encoder_class" in trove_conf:
                args.encoder_class = trove_conf["encoder_class"]

        # separate kwargs that are only used for BiEncoderRetriever.__init__()
        cls_kwargs = dict()
        for k in ["format_query", "format_passage", "append_eos_token"]:
            if k in kwargs:
                cls_kwargs[k] = kwargs.pop(k)
        # this one is used by both retriever and encoder init methods.
        # So, do not '.pop' it from kwargs
        if "preprocess_only" in kwargs:
            cls_kwargs["preprocess_only"] = kwargs["preprocess_only"]

        encoder: PretrainedEncoder = PretrainedEncoder.from_model_args(
            args=args,
            model_name_or_path=model_name_or_path,
            training_args=training_args,
            **kwargs,
        )

        retriever = cls(
            model_args=args,
            encoder=encoder,
            loss_extra_kwargs=loss_extra_kwargs,
            **cls_kwargs,
        )
        return retriever

    def forward(
        self,
        query: Optional[Dict[str, torch.Tensor]] = None,
        passage: Optional[Dict[str, torch.Tensor]] = None,
        label: Optional[torch.Tensor] = None,
        return_loss: Optional[bool] = True,
        **kwargs,
    ) -> RetrieverOutput:
        """Encodes query and passages and potentially calculate similarity scores and loss.

        Args:
            query (Optional[Dict[str, torch.Tensor]]): tokenized query.
            passage (Optional[Dict[str, torch.Tensor]]): tokenized passages.
            label (Optional[torch.Tensor]): Relevancy scores of the corresponding passages for each query.
                If there are k documents for each query, this is a 2D tensor of shape `[NUM_QUERIES, k]`
            return_loss (Optional[bool]): if true, calculate the loss value.
            **kwargs: unused keyword arguments are passed to the ``forward()`` method of the loss module.

        Returns:
            If possible, query and passage embeddings as well as similarity and loss scores.
        """
        raise NotImplementedError

    def save_pretrained(self, *args, **kwargs):
        is_main_process = True
        if "is_main_process" in kwargs:
            is_main_process = kwargs["is_main_process"]
        elif len(args) > 1:
            is_main_process = args[1]
        if is_main_process:
            if "save_directory" in kwargs:
                pardir = Path(kwargs["save_directory"])
            else:
                pardir = Path(args[0])
            pardir.mkdir(exist_ok=True, parents=True)
            conf = {"encoder_class": self.encoder.__class__.__name__}
            with open(pardir / modeling_utils.TROVE_RETRIEVER_CONFIG_FILE, "w") as f:
                json.dump(conf, f, indent=2)

        if "state_dict" in kwargs:
            encoder_prefix = "encoder."
            loss_prefix = "loss."
            assert all(
                k.startswith(encoder_prefix) or k.startswith(loss_prefix)
                for k in kwargs["state_dict"].keys()
            )
            kwargs["state_dict"] = {
                k[len(encoder_prefix) :]: v
                for k, v in kwargs["state_dict"].items()
                if not k.startswith(loss_prefix)
            }
        return self.encoder.save_pretrained(*args, **kwargs)
