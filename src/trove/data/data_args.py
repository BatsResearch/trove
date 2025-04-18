"""Configs for loading and processing the data.

From here:
https://github.com/texttron/tevatron/blob/7d298b428234f1c1065e98244827824753361815/src/tevatron/retriever/arguments.py
"""

from dataclasses import dataclass, fields
from typing import Dict, Optional


@dataclass
class DataArguments:

    dataset_name: Optional[str] = None
    """Name of the dataset. Only used if your query/passage formatting functions behave differently for different datasets"""
    group_size: int = 8
    """Number of passages used for each query during training or approximate evaluation during training
    (i.e., only used with RetrievalTrainer and NOT used with RetrievalEvaluator).
    """
    positive_passage_no_shuffle: bool = False
    """(for binary IR dataset) always use the first positive passage for training"""
    negative_passage_no_shuffle: bool = False
    """(for binary IR dataset) always use the first n negative passages for training"""
    passage_selection_strategy: str = "most_relevant"
    """(Only for MultiLevelDataset) How to choose a subset of passages for each query.
    Valid options are None, 'random', 'least_relevant', and 'most_relevant'.
    """
    query_max_len: Optional[int] = 32
    """The maximum total input sequence length after tokenization for query. Sequences longer
    than this will be truncated, sequences shorter will be padded.
    """
    passage_max_len: Optional[int] = 128
    """The maximum total input sequence length after tokenization for passage. Sequences longer
    than this will be truncated, sequences shorter will be padded.
    """
    pad_to_multiple_of: Optional[int] = 16
    """If set will pad the sequence to a multiple of the provided value. This is especially useful to
    enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
    """

    def to_dict(self) -> Dict:
        """Return a json serializable view of the class attributes."""
        json_dict = dict()
        field_names = [f.name for f in fields(self)]
        for fname in field_names:
            fvalue = getattr(self, fname)
            json_dict[fname] = fvalue
        return json_dict
