"""Data collator for information retrieval."""

import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from ..logging_utils import get_logger_with_config
from .data_args import DataArguments

logger, logging_conf = get_logger_with_config("trove")


def get_tokenized(
    inputs: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    append_eos: bool,
    pad_to_multiple_of: Optional[int],
) -> Dict[str, torch.Tensor]:
    """Tokenize the input texts.

    From here:
    https://github.com/texttron/tevatron/blob/7d298b428234f1c1065e98244827824753361815/src/tevatron/retriever/collator.py

    Args:
        inputs (List[str]): list of input texts.
        tokenizer (PreTrainedTokenizer): transformers tokenizer
        max_length (int): max length of the result
        append_eos (bool): whether to append the ``eos`` token to ``input_ids``
        pad_to_multiple_of (Optional[int]): ``pad_to_multiple_of`` argument of the tokenizer

    Returns:
        tokenized inputs in pytorch tensors.
    """

    output = tokenizer(
        inputs,
        padding=False,
        truncation=True,
        max_length=max_length - 1 if append_eos else max_length,
        return_attention_mask=False,
        return_token_type_ids=False,
        add_special_tokens=True,
    )

    if append_eos:
        output["input_ids"] = [
            i + [tokenizer.eos_token_id] for i in output["input_ids"]
        ]

    output = tokenizer.pad(
        output,
        padding=True,
        pad_to_multiple_of=pad_to_multiple_of,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return output


@dataclass
class RetrievalCollator:
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer
    append_eos: bool

    def __post_init__(self):
        # keys are names of the fields in the output of the dataset (e.g., passage, label, etc.)
        # values are a boolean that is 'True' if the field can be converted to a torch tensor and false otherwise
        # When we face a field that we do not recognize, we first try to convert it to a tensor and if we
        # succeed, we set the corresponding value to 'True' and false otherwise.
        # the next time that we encounter the field, it is faster to decide whether it should be converted to tensor or not
        self.field_is_tensor = dict()

    def __call__(
        self, features: List[Dict[str, Union[str, List[str], List[Union[int, float]]]]]
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """tokenize and collate a list of examples.

        Args:
            features (List[Dict[str, Union[str, List[str], List[int]]]]): A list of examples (i.e., outputs of the dataset `__getitem__` function).
                We only process these keys in each example dictionary: ``['query', 'passage', 'label']``

                * ``'query'``: a text string
                * ``'passage'``: a list of text strings
                * ``'label'``: a list of integers or floats

                If there are additional keys, their values are collected in a list and returned as is.

        Returns:
            a mapping with collated ``'query'`` and ``'passage'`` tokens if available.
            If available in input features, ``'label'`` are also returned as a torch tensor.
        """
        features_dict = defaultdict(list)

        for example in features:
            for key, value in example.items():
                features_dict[key].append(value)

        output = dict()

        if len(features_dict["query"]) != 0:
            output["query"] = get_tokenized(
                inputs=features_dict["query"],
                tokenizer=self.tokenizer,
                max_length=self.data_args.query_max_len,
                append_eos=self.append_eos,
                pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            )
        if len(features_dict["passage"]) != 0:
            if isinstance(features_dict["passage"][0], list):
                features_dict["passage"] = list(
                    itertools.chain.from_iterable(features_dict["passage"])
                )
            output["passage"] = get_tokenized(
                inputs=features_dict["passage"],
                tokenizer=self.tokenizer,
                max_length=self.data_args.passage_max_len,
                append_eos=self.append_eos,
                pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            )
        if len(features_dict["embedding"]) != 0:
            output["embedding"] = torch.tensor(np.array(features_dict["embedding"]))
        if len(features_dict["label"]) != 0:
            output["label"] = torch.tensor(features_dict["label"])

        _already_checked = ["query", "passage", "label", "embedding"]
        for feat_name, feat_values in features_dict.items():
            if feat_name in _already_checked or len(feat_values) == 0:
                continue
            if feat_name not in self.field_is_tensor:
                try:
                    torch.tensor(feat_values)
                    self.field_is_tensor[feat_name] = True
                except:
                    self.field_is_tensor[feat_name] = False
            if self.field_is_tensor[feat_name]:
                output[feat_name] = torch.tensor(feat_values)
            else:
                output[feat_name] = feat_values

        return output
