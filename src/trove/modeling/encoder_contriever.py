from typing import Dict, Optional

import torch
from transformers import AutoModel, PreTrainedModel, TrainingArguments

from . import modeling_utils
from .model_args import ModelArguments
from .pretrained_encoder import PretrainedEncoder


class ContrieverEncoder(PretrainedEncoder):
    @classmethod
    def can_wrap(cls, model_name_or_path: str, args: ModelArguments) -> bool:
        """Returns true if the model is an instance of ``facebook/contriever`` or
        ``facebook/contriever-msmarco`` models.

        We can also wrap models that their base model is one of these. But, :class:`~trove.modeling.pretrained_encoder.PretrainedEncoder` takes
        care of that. We just handle the main models.
        """
        _can_wrap = model_name_or_path in [
            "facebook/contriever",
            "facebook/contriever-msmarco",
        ]
        return _can_wrap

    def __init__(
        self,
        args: ModelArguments,
        training_args: Optional[TrainingArguments] = None,
        preprocess_only: bool = False,
        **kwargs,
    ) -> None:
        """Wraps contriever variants and also provides necessary attributes and methods for data
        pre-processing.

        Args:
            args (ModelArguments): config for instantiating the model
            training_args (TrainingArguments): Not used by this wrapper.
            preprocess_only (bool): If True, do not load model parameteres and only
                provide methods and attributes necessary for pre-processing the input data.
            **kwargs: passed to ``transformers.AutoModel.from_pretrained``.
        """
        super().__init__(args, **kwargs)

        if not preprocess_only:
            if self.args.use_peft or self.args.load_in_4bit or self.args.load_in_8bit:
                raise NotImplementedError

            torch_dtype = (
                self.args.torch_dtype
                if self.args.torch_dtype in ["auto", None]
                else getattr(torch, self.args.torch_dtype)
            )
            model_kwargs = dict(
                revision=self.args.model_revision,
                trust_remote_code=self.args.trust_remote_code,
                torch_dtype=torch_dtype,
            )
            self.model: PreTrainedModel = AutoModel.from_pretrained(
                self.args.model_name_or_path, **model_kwargs, **kwargs
            )
            # The wrapped model should take care of these
            # Expose these to remain compatible with a subset (and *NOT* all) of tools
            # (e.g., transformers.Trainer) that expect a transformers.PretrainedModel instance
            modeling_utils.add_model_apis_to_wrapper(wrapper=self, model=self.model)
        else:
            self.model = None

        self.append_eos_token = False

    def encode(self, inputs: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        outputs = self.model(**inputs)
        token_embeddings = outputs[0]
        mask = inputs["attention_mask"]
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def format_query(self, text: str, **kwargs) -> str:
        return text.strip()

    def format_passage(self, text: str, title: Optional[str] = None, **kwargs) -> str:
        if title is None:
            return text.strip()
        else:
            return f"{title.strip()} {text.strip()}".strip()

    def save_pretrained(self, *args, **kwargs):
        if "state_dict" in kwargs:
            prefix = "model."
            assert all(k.startswith(prefix) for k in kwargs["state_dict"].keys())
            kwargs["state_dict"] = {
                k[len(prefix) :]: v for k, v in kwargs["state_dict"].items()
            }
        return self.model.save_pretrained(*args, **kwargs)
