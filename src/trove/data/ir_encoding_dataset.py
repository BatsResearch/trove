import copy
import itertools
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import datasets
import numpy as np
from torch.utils.data import Dataset

from ..logging_utils import get_logger_with_config
from .data_args import DataArguments
from .vector_cache_mixin import VectorCacheMixin

logger, logging_conf = get_logger_with_config("trove")


class EncodingDataset(VectorCacheMixin, Dataset):
    def __init__(
        self,
        data_args: Optional[DataArguments] = None,
        data_args_overrides: Optional[Dict[str, Any]] = None,
        dataset_name: Optional[str] = None,
        query_path: Optional[os.PathLike] = None,
        corpus_path: Optional[os.PathLike] = None,
        format_query: Optional[Callable[[str, Optional[str]], str]] = None,
        format_passage: Optional[Callable[[str, str, Optional[str]], str]] = None,
        global_id_suffix: Optional[str] = None,
        cache_path: Optional[os.PathLike] = None,
        load_cache_on_init: bool = True,
        prefer_cache: bool = True,
    ) -> None:
        """Dataset for encoding query and passage texts.

        **This class supports reading/writing dense embedding vectors to cache on disk.**

        Args:
            data_args (Optional[DataArguments]): general arguments for loading and processing the data.
                Currently only used to find out ``dataset_name`` if not explicitely provided.
                And that is only useful if your query/passage formatting functions require a ``dataset`` argument.
            data_args_overrides (Optional[Dict[str, Any]]): A mapping from a subset of ``DataArguments`` attribute names to
                their new values. These key values override the corresponding attributes in ``data_args`` argument.
                It is useful if you want to create multiple datasets from the same ``DataArguments`` instance
                but make small changes for each dataset without creating new ``DataArguments`` instances.
            dataset_name (Optional[str]): Name of the dataset that is being encoded.
                only useful if your query/passage formatting functions require a ``dataset`` argument.
            query_path (Optional[os.PathLike]): Path to JSONL file containing query texts.
            corpus_path (Optional[os.PathLike]): Path to JSONL file containing passage texts.
            format_query (Optional[Callable]): callable that takes query text and dataset name and returns the formatted query text.
            format_passage (Optional[Callable]): callable that takes passage text and title and dataset name and returns the formatted passage text.
            global_id_suffix (Optional[str]): unique file id used to create globally unique query/passage ids across files.
            cache_path (Optional[os.PathLike]): path to cache file
            load_cache_on_init (bool): if true and ``cache_path`` is provided and exists, load the cache arrow table at the end of init method
            prefer_cache (bool): If true, use cache if present. If false, ignore cache even if it exists.
        """
        super().__init__(cache_file_name=cache_path)

        # We only support one source of data
        if ((query_path is None) == (corpus_path is None)) or (
            (format_query is None) == (format_passage is None)
        ):
            msg = (
                "We only support encoding either queries or documents in each instance of 'EncodingDataset'."
                " You should provide one and only one of the ('query_path', 'format_query')"
                " or ('corpus_path', 'format_passage') argument pairs."
                f" Got: 'query_path': '{query_path}', 'format_query': '{format_query}',"
                f" 'corpus_path': '{corpus_path}', 'format_passage': '{format_passage}'"
            )
            raise ValueError(msg)

        # make a deepcopy so we do not impact operations outside of this class.
        data_args = copy.deepcopy(data_args)
        if data_args_overrides is not None:
            for attr, val in data_args_overrides.items():
                if not hasattr(data_args, attr):
                    # We should not add new attributes
                    msg = f"'{attr}' is not a valid attribute name for 'DataArguments'"
                    raise ValueError(msg)
                setattr(data_args, attr, val)
        self.args = data_args

        self.dataset_name = None
        if dataset_name is not None:
            self.dataset_name = dataset_name
        elif self.args is not None and self.args.dataset_name is not None:
            self.dataset_name = self.args.dataset_name

        self._prefer_cache = prefer_cache
        self.format_query = format_query
        self.format_passage = format_passage

        # See `MaterializedQRel.local_to_global_id()` for more info
        if global_id_suffix is not None:
            self.global_id_suffix = "_" + global_id_suffix
        else:
            self.global_id_suffix = ""

        if query_path is not None:
            self.filepath = query_path
            self.is_query = True

        if corpus_path is not None:
            self.filepath = corpus_path
            self.is_query = False

        if Path(self.filepath).suffix not in [".json", ".jsonl"]:
            msg = "Only json/jsonl files are supported"
            raise NotImplementedError(msg)

        logger.info(f"Load Encoding data from file:\n{self.filepath}")
        self.dataset = datasets.load_dataset(
            "json", data_files=self.filepath, split="train"
        )
        # Index of all rows that this dataset can access.
        self.all_row_indices = list(range(len(self.dataset)))
        # Index of rows in the current shard.
        self.shard_row_indices = list(range(len(self.dataset)))
        self.shard_idx = None
        self.num_shards = None

        # Find the field that contains the query/passage ID
        ds_fields = list(self.dataset[0].keys())
        all_id_fields = [
            "_id",
            "qid",
            "query_id",
            "query-id",
            "docid",
            "passage_id",
            "passage-id",
            "corpus-id",
        ]
        ds_id_fields = list(set(ds_fields).intersection(set(all_id_fields)))
        assert len(ds_id_fields) == 1
        self.id_field = ds_id_fields[0]

        # Unique ID of all the available records in the entire dataset regardless of soft sharding
        self._all_rec_ids = self.dataset[self.id_field]

        if load_cache_on_init:
            self.load_cache()

    @property
    def all_rec_ids(self) -> List[str]:
        return self._all_rec_ids

    def ignore_cache(self) -> None:
        """Return raw data even if cache exists."""
        self._prefer_cache = False

    def prefer_cache(self) -> None:
        """Return cached embeddings if possible."""
        self._prefer_cache = True

    @contextmanager
    def disable_cache(self):
        """Context manager to temporarily disable vector cache."""
        orig_state = self._prefer_cache
        self.ignore_cache()
        try:
            yield
        finally:
            self._prefer_cache = orig_state

    def shard(
        self,
        shard_idx: int,
        num_shards: Optional[int] = None,
        shard_weights: Optional[List[float]] = None,
        hard_shard: bool = False,
    ) -> None:
        """Shard the dataset.

        It can do both a `hard` and `soft` shard.

        In a soft shard, we only mask the index of rows that are not in the shard.
        And, the underlying data is not changed and you can reverse the sharding by calling the ``unshard()`` method.

        In a hard shard, however, the underlying data is changed and the effects of sharding are irreversible for this instance.
        Also keep in mind that after a hard shard, the dataset will seem unsharded to other methods/classes like ``VectorCacheMixin``.

        In most cases, you should just use the default soft sharding.
        Hard sharding is mostly useful if you plan to shard the dataset twice.
        For example, if you want to encode each shard in a separate session which itself runs in a distributed environment,
        You can do a hard shard in each session before passing the dataset to ``trove.RetrievalEvaluator()``, for instance.
        This allows ``RetrievalEvaluator`` to further shard the dataset in each session into smaller pieces for each process.

        Args:
            shard_idx (int): Index of current shards
            num_shards (Optional[int]): Total number of shards. If ``shard_weights`` is
                provided, ``num_shards`` should be either ``None`` or equal to ``len(shard_weights)``.
            shard_weights (Optional[List[float]]): relative number of items in each shard.
                If provided, shard the dataset according to these weights. If not provided, all shards
                are of the same size.
            hard_shard (bool): Whether to a hard/permanent shard or a soft/superficial shard.
        """
        if shard_weights is None and num_shards is None:
            msg = "You must provide at least one of 'shard_weights' or 'num_shards'. Got None for both."
            raise ValueError(msg)
        if (
            shard_weights is not None
            and num_shards is not None
            and len(shard_weights) != num_shards
        ):
            msg = "'num_shards' is given but is not equal to 'len(shard_weights)'"
            raise ValueError(msg)

        if self.shard_idx is not None or self.num_shards is not None:
            msg = (
                "Nested soft sharding is not supported."
                " Nested sharding leads to similar cache file names for different subsets of data."
                " You might want to do a hard shard followed by a soft shard to achieve similar results."
            )
            raise NotImplementedError(msg)

        # Make sure it is ok to shard the dataset from the cache perspective.
        # E.g. You cannot shard dataset if cache file is open for writing.
        self._shard_checks_by_cache()

        num_items = len(self.all_row_indices)
        if shard_weights is None:
            # Chunk the dataset into shards of equal sizes
            shard_size = math.ceil(num_items / num_shards)
            shard_boundaries = [shard_idx * shard_size, (shard_idx + 1) * shard_size]
        else:
            # The size of each shard is determined by shard_weights
            num_shards = len(shard_weights)
            norm_shard_weights = [i / sum(shard_weights) for i in shard_weights]
            shard_sizes = [math.ceil(num_items * i) for i in norm_shard_weights]
            all_shard_boundaries = list(itertools.accumulate([0] + shard_sizes))
            assert all_shard_boundaries[-1] >= num_items
            all_shard_boundaries = list(itertools.pairwise(all_shard_boundaries))
            shard_boundaries = all_shard_boundaries[shard_idx]

        self.shard_row_indices = self.all_row_indices[
            shard_boundaries[0] : shard_boundaries[1]
        ]

        self.shard_idx = shard_idx
        self.num_shards = num_shards

        if hard_shard:
            # Forget everything about other shards.
            # Make it look like this shard is the only thing in the dataset
            # Make sure to use shallow copy when updating indices.
            self.all_row_indices = self.shard_row_indices[:]
            self._all_rec_ids = self.dataset.select(self.all_row_indices)[self.id_field]
            # Since in a hard shard, the dataset looks as unsharded to other classes including `VectorCacheMixin`,
            # We have to manually separate the cache file for this shard from that of other shards.
            # We do it by saving all vector cache files related to this shard in a separate subdirectory
            # in the directory that unsharded cache files would have been saved
            width = len(str(int(self.num_shards))) + 1
            cache_subdir = (
                f"hard_shard-{self.shard_idx:0{width}}-of-{self.num_shards:0{width}}"
            )
            # Make it look like unsharded dataset
            self.shard_idx = None
            self.num_shards = None
            # Let VectorCacheMixin know that something permanent has changed about this dataset
            self.update_cache_subdir(subdir=cache_subdir, append=True, load=False)

    def unshard(self) -> None:
        """Reverse the sharding.

        I.e., unmask the index of rows accessible to dataset
        """
        # Make a shallow copy and do not assign directly
        self.shard_row_indices = self.all_row_indices[:]
        self.shard_idx = None
        self.num_shards = None

    def __len__(self) -> int:
        # Just consider the rows that we are allowed to access
        return len(self.shard_row_indices)

    def __getitem__(self, index: int) -> Dict[str, Union[str, np.ndarray]]:
        """Return either the formatted raw data or cached embeddings."""
        rec = self.dataset[self.shard_row_indices[index]]
        rec_id = rec[self.id_field]  # Unique query/document ID

        output = dict()
        emb = None
        if self._prefer_cache:
            emb = self.get_cached_value(rec_id)

        if emb is not None:
            # only return cached value if found/exists
            output["embedding"] = emb
        else:
            if self.is_query:
                output["query"] = self.format_query(
                    text=rec["text"], dataset=self.dataset_name
                )
            else:
                output["passage"] = self.format_passage(
                    text=rec["text"], title=rec["title"], dataset=self.dataset_name
                )

        # The original record ID and the global record ID used to allow
        # merging these records with records from other files.
        # See `MaterializedQRel.local_to_global_id()` for more info
        # **You should always use `orig_rec_id` to read/write from/to cache.
        output["orig_rec_id"] = rec_id
        output["rec_id"] = rec_id + self.global_id_suffix

        return output
