"""This file contains base wrapper for encoder architectures.

See 'PretrainedEncoder' docstring for the primary goal of these wrapper classes and how to add a
new one.
"""

from typing import Dict, Optional

import torch
from torch import nn
from transformers import TrainingArguments

from . import modeling_utils
from .model_args import ModelArguments


class PretrainedEncoder(nn.Module):
    """A wrapper around different encoders.

    This class wraps the encoder and takes care of model specific actions like
    saving and loading checkpoints, pooling, normalization, formatting inputs, etc.

    ``PretrainedEncoder`` automatically detects and instantiates the correct wrapper subclass that can load a specific model.

    To support a new type of encoder, inherit from this class and implement the class method
    ``cls.can_wrap()``. This method takes the model name and its arguments and
    if it can wrap this model, returns True. Otherwise, returns False.

    **Do not overwrite or modify** ``cls._model_registry``.
    """

    _model_registry = list()
    _alias: Optional[str] = None

    def __init_subclass__(cls, **kwargs):
        """Store a pointer to new wrappers that inherit from `PretrainedEncoder`."""
        super().__init_subclass__(**kwargs)
        # prioritize user-defined wrappers over the ones defined by the library.
        if "trove" in cls.__module__.split("."):
            cls._model_registry.append(cls)
        else:
            cls._model_registry.insert(0, cls)

    @classmethod
    def can_wrap(cls, model_name_or_path: str, args: ModelArguments) -> bool:
        """returns true if this wrapper can wrap the specified model with the given arguments.

        Subclasses of ``PretrainedEncoder`` should implement this method.
        We use this method to automatically choose the correct subclass to wrap different models.

        Args:
            model_name_or_path (str): name of the model to wrap.
            args (ModelArguments): arguments that describe the model to wrap.

        Returns:
            True if this class can wrap the model, and false otherwise.
        """
        raise NotImplementedError

    @classmethod
    def find_appropriate_wrapper(cls, model_name_or_path: str, args: ModelArguments):
        """Find the appropriate wrapper than can wrap and load a specific model.

        If ``args.encoder_class`` is set, then a subclass of ``PretrainedEncoder`` with that name (or alias) is returned.
        Otherwise, the model arguments are passed to the ``can_wrap()`` method of all the registered subclasses of ``PretrainedEncoder``.
        The subclass that its ``can_wrap()`` method returns ``True`` is returned by this method.
        If the ``can_wrap()`` method of all subclasses return ``False`` and the checkpoint is a fine-tuned model, it finds the base model of the given checkpoint and
        repeats the same process to find the subclass that can load the base model.

        Args:
            model_name_or_path (str): name or path of the model that we want to wrap.
            args (ModelArguments): arguments that describe the model that we want to wrap.
                ``cls.can_wrap()`` method of subclasses might use this in addition
                to ``model_name_or_path`` to determine if they can wrap the model.

        Returns:
            A pointer to the subclass that can wrap the given model.
        """
        if args.encoder_class is not None and args.encoder_class.lower() != "none":
            # find the encoder wrapper with the given name
            encoder_cls = None
            for model_cls in cls._model_registry:
                if (
                    model_cls.__name__ == args.encoder_class
                    or model_cls._alias == args.encoder_class
                ):
                    encoder_cls = model_cls
                    break
            if encoder_cls is None:
                msg = f"Could not find a PretrainedEncoder subclass with the name or alias equal to '{args.encoder_class}'"
                raise ValueError(msg)
            return encoder_cls

        assert model_name_or_path is not None

        # List of classes that can load the model
        compatible_classes = list()
        for model_cls in cls._model_registry:
            if model_cls.can_wrap(model_name_or_path=model_name_or_path, args=args):
                compatible_classes.append(model_cls)
        if len(compatible_classes) > 1:
            msg = (
                "More than one wrapper can load this model. We cannot decide which one to use."
                f" Compatible classes: {compatible_classes}"
            )
            raise RuntimeError(msg)

        if len(compatible_classes) == 1:
            return compatible_classes[0]

        # None of the wrappers can load this model
        # Try to use the wrapper that can load the base model of the specified model.
        base_model_name = modeling_utils.find_base_name_or_path(model_name_or_path)
        if base_model_name is None:
            msg = "None of the encoder wrappers can load this model or any of its base models."
            raise RuntimeError(msg)

        encoder_cls = cls.find_appropriate_wrapper(
            model_name_or_path=base_model_name, args=args
        )
        return encoder_cls

    @classmethod
    def from_model_args(
        cls,
        args: ModelArguments,
        model_name_or_path: Optional[str] = None,
        training_args: Optional[TrainingArguments] = None,
        **kwargs,
    ):
        """Instantiate the correct subclass of ``PretrainedEncoder`` that can wrap the specified
        model.

        Args:
            args (ModelArguments): arguments that describe the model that we want to wrap.
            model_name_or_path (Optional[str]): name of the model to wrap.
                If not provided, use ``args.model_name_or_path``.
            training_args (TrainingArguments): passed to `__init__`. Used for activating gradient checkpointing.
            **kwargs: extra keyword arguments passed to the wrapper constructor.

        Returns:
            an instance of the correct wrapper class that wraps the specified model.
        """
        if model_name_or_path is None:
            model_name_or_path = args.model_name_or_path
        encoder_cls = cls.find_appropriate_wrapper(
            model_name_or_path=model_name_or_path, args=args
        )
        return encoder_cls(args=args, training_args=training_args, **kwargs)

    def __init__(
        self,
        args: ModelArguments,
        training_args: Optional[TrainingArguments] = None,
        preprocess_only: bool = False,
        **kwargs,
    ) -> None:
        """Wraps encoder models and provides methods and attributes for pre-processing encoder
        inputs.

        If ``preprocess_only`` is True, you are expected to only provide attributes and methods
        required for preparing the input for the encoder (e.g., ``append_eos_token``, ``format_query()``, etc.).
        If possible you should avoid loading the model parameters if ``preprocess_only`` is True.
        This allows us to pre-process the data without loading the model, which leaves
        more resources (e.g., memory) for preprocessing operations.

        Args:
            args (ModelArguments): config for instantiating the model
            preprocess_only (bool): if true, do not load the model parameters
                and just provide the attributes and methods necessary for pre-processing the input.
                E.g., ``append_eos_token``, ``format_query()``, etc.
            training_args (TrainingArguments): You can use this to activate gradient checkpointing if needed.
            **kwargs: extra kwargs passed to PretrainedModel.from_pretrained when loading
                the encoder model.
        """
        super().__init__()
        self.args = args
        self.config = None  # You should set this to config of the encoder (PretrainedModel.config)

    def encode(self, inputs: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """calculate the embeddings for tokenized input."""
        raise NotImplementedError

    def format_query(self, text: str, **kwargs) -> str:
        """Format the query before passing it to tokenizer.

        You can also ask for other parameters like dataset_name for example if your model uses
        different formatting for different datasets like `intfloat/e5-mistral-7b-instruct`
        """
        raise NotImplementedError

    def format_passage(
        self,
        text: str,
        title: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Format the passage before passing it to tokenizer.

        You can also ask for other parameters like dataset_name for example if your model uses
        different formatting for different datasets like `intfloat/e5-mistral-7b-instruct`
        """
        raise NotImplementedError

    def save_pretrained(self, *args, **kwargs):
        """Save model parameters.

        It should replicate the signature and behavior of
        ``transformers.PreTrainedModel.save_pretrained``.
        """
        raise NotImplementedError

    def encode_query(self, inputs: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Overwrite if queries are encoded differently."""
        return self.encode(inputs=inputs)

    def encode_passage(self, inputs: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Overwrite if passages are encoded differently."""
        return self.encode(inputs=inputs)

    def similarity_fn(self, query: torch.Tensor, passage: torch.Tensor) -> torch.Tensor:
        """Similarity between query and passage embeddings.

        Overwrite if your encoder uses a different similarity function.

        Args:
            query (torch.Tensor): query embeddings. shape: `[NUM_QUERIES, EMB_DIM]`
            passage (torch.Tensor): passage embeddings. shape: `[NUM_PASSAGES, EMB_DIM]`

        Returns:
            query-passage similarities. shape is `[NUM_QUERIES, NUM_PASSAGES]`
        """
        return query @ passage.T

    def compute_scores(
        self, query: Dict[str, torch.Tensor], passage: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute similarity score between tokenized query and passages.

        Args:
            query (Dict[str, torch.Tensor]): query tokens
            passage (Dict[str, torch.Tensor]): passage tokens

        Returns:
            query-passage similarities. shape is `[NUM_QUERIES, NUM_PASSAGES]`
        """
        query_feats = self.encode_query(inputs=query)
        passage_feats = self.encode_passage(inputs=passage)
        scores = self.similarity_fn(queries=query_feats, passage=passage_feats)
        return scores

    def gradient_checkpointing_enable(self, *args, **kwargs):
        return self.model.gradient_checkpointing_enable(*args, **kwargs)
