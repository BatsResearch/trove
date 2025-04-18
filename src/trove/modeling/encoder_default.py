from typing import Dict, Optional

import torch
from transformers import TrainingArguments

from . import modeling_utils
from .model_args import ModelArguments
from .pretrained_encoder import PretrainedEncoder


class DefaultEncoder(PretrainedEncoder):
    _alias = "default"

    @classmethod
    def can_wrap(cls, *args, **kwargs) -> bool:
        return False

    def __init__(
        self,
        args: ModelArguments,
        training_args: Optional[TrainingArguments] = None,
        preprocess_only: bool = False,
        **kwargs,
    ) -> None:
        """A generic wrapper that can be used with any model.

        You can customize the pooling and normalization through ``model_args`` options.
        This wrapper does not do any custom formatting for queries and passages.
        It just returns the main text (prefix by the title if available).

        Args:
            args (ModelArguments): config for instantiating the model
            training_args (TrainingArguments): Not used by this wrapper.
            preprocess_only (bool): If True, do not load model parameteres and only
                provide methods and attributes necessary for pre-processing the input data.
            **kwargs: passed to ``transformers.AutoModel.from_pretrained``.
        """
        super().__init__(args, **kwargs)

        if args.normalize is None or args.pooling is None:
            msg = "You should specify both `normalize` and `pooling` options when using the default encoder wrapper."
            raise ValueError(msg)

        if args.normalize.lower() == "yes":
            self.normalize = True
        elif args.normalize.lower() == "no":
            self.normalize = False
        else:
            msg = f"'normalize' option must be one of `yes` or `no`. Got '{args.normalize}'"
            raise ValueError(msg)

        self.append_eos_token = False
        if args.pooling.lower() == "mean":
            self.pooling_fn = modeling_utils.mean_pooling
        elif args.pooling.lower() == "first_token":
            self.pooling_fn = modeling_utils.first_token_pooling
        elif args.pooling.lower() == "last_token":
            self.pooling_fn = modeling_utils.last_token_pooling
            self.append_eos_token = True
        else:
            msg = f"'pooling' option must be one of `mean`, `first_token`, or `last_token`. Got '{args.pooling}'"
            raise ValueError(msg)

        if not preprocess_only:
            self.model = modeling_utils.load_transformers_model(
                model_args=args, training_args=training_args, **kwargs
            )
            # The wrapped model should take care of these
            # Expose these to remain compatible with a subset (and *NOT* all) of tools
            # (e.g., transformers.Trainer) that expect a transformers.PretrainedModel instance
            modeling_utils.add_model_apis_to_wrapper(wrapper=self, model=self.model)
        else:
            self.model = None

    def encode(self, inputs: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        hidden_states = self.model(**inputs, return_dict=True)
        hidden_states = hidden_states.last_hidden_state
        att_mask = inputs["attention_mask"]

        embs = self.pooling_fn(last_hidden_state=hidden_states, attention_mask=att_mask)
        if self.normalize:
            embs = torch.nn.functional.normalize(embs, p=2, dim=-1)
        return embs

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
