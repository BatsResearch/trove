import os
import pickle as pkl
from pathlib import Path
from typing import Any, Dict, List, Optional

import datasets
from tqdm import tqdm

from .. import cache_manager, file_utils
from ..logging_utils import get_logger_with_config
from . import container_utils
from .lmdb_dict import LMDBDict

logger, logging_conf = get_logger_with_config("trove")


def encode_key(key: str) -> bytes:
    """Encode str keys as bytes for saving to LMDBDict."""
    return key.encode("utf-8")


def decode_key(key: bytes) -> str:
    """Decode keys read from LMDBDict from bytes to str."""
    return key.decode("utf-8")


def encode_value(value: List[int]) -> bytes:
    """Encode values to bytes using python pickle."""
    return pkl.dumps(value)


def decode_value(value: bytes) -> List[int]:
    """Decode values from bytes to original objects using python pickle."""
    return pkl.loads(value)


def lmdb_key_to_row_indices_from_dict(
    db_file: os.PathLike,
    db_hash_file: Optional[os.PathLike] = None,
    k2r_idx: Optional[Dict[str, List[int]]] = None,
    k2r_idx_file: Optional[os.PathLike] = None,
) -> LMDBDict:
    if (k2r_idx is None) == (k2r_idx_file is None):
        msg = (
            "You should pass exactly one of the 'k2r_idx' or 'k2r_idx_file' arguments."
            f" Got: type(k2r_idx): '{type(k2r_idx)}' and type(k2r_idx_file): '{type(k2r_idx_file)}'"
        )
        raise ValueError(msg)

    if db_hash_file is not None:
        db_hash_file = Path(db_hash_file)

    with file_utils.easyfilelock(db_file):
        # only process should write to db file. See LMDBDict docstrings for details.
        if not Path(db_file).exists():
            if k2r_idx is None:
                with open(k2r_idx_file, "rb") as f:
                    k2r_idx = pkl.load(f)
            hash_content = LMDBDict.create_db(
                data=k2r_idx.items(),
                db_file=db_file,
                encode_key=encode_key,
                encode_value=encode_value,
            )
            db_hash_file.parent.mkdir(exist_ok=True, parents=True)
            db_hash_file.write_text(hash_content)

    _hash = None
    if db_hash_file is not None and db_hash_file.exists():
        _hash = db_hash_file.read_text().strip()
    memmap_k2r_idx = LMDBDict.load_db(
        db_file=db_file,
        db_hash=_hash,
        encode_key=encode_key,
        decode_key=decode_key,
        encode_value=encode_value,
        decode_value=decode_value,
    )
    return memmap_k2r_idx


class KeyToRowIndices:
    def __init__(
        self,
        dataset: datasets.Dataset,
        id_field: str,
        storage: str = "in_memory",
        num_proc: Optional[int] = None,
    ) -> None:
        """Dict from key values to row indices for HF datasets.

        This class allows you to access HF dataset rows with the value of a specific column.

        It can use a memory-mapped dictionary implemented using ``lmdb`` to reduce memory usage.
        You do not need to use memory mapping. You should be fine with regular dictionary even for tens of millions of records.

        Example:

        .. code-block:: python

            ds = datasets.Dataset.from_dict({'_id': ['a', 'b'], 'val': ['foo', 'bar']})
            k2ridx = KeyToRowIndices(ds, id_field='_id')
            assert k2ridx['a'] == [0] # list of index rows that their '_id' field is equal to 'a'
            assert [ds[i] for i in k2ridx['a']] == [{'_id': 'a', 'val': 'foo'}]

        This class provides a subset of python dictionary API.

        Args:
            dataset (datasets.Dataset): dataset to create index mapping for.
            id_field (str): name of the field to use as key.
            storage (str): whether load the key to row indices mapping in memory or use
                a memory mapped dict (using ``lmdb``).
            num_proc (Optional[int]): arg to `datasets.Dataset.*` methods.
        """
        logger.debug(
            f"Create memory mapped key to row indices mapping. num records: {tqdm.format_sizeof(len(dataset))}"
        )

        if storage not in ["in_memory", "memory_mapped"]:
            msg = f"valid values for 'storage' argument are 'in_memory' and 'memory_mapped'. Got: '{storage}'"
            raise ValueError(msg)

        self.k2r_idx_pkl_file = container_utils.dataset_column_to_row_indices_mapping(
            dataset=dataset, id_field=id_field, num_proc=num_proc, load=False
        )
        db_pardir = cache_manager.get_cache_dir(
            input_data=dataset,
            artifact_type="final",
            artifact_content="lmdb_dict_key_to_row_indices_mapping",
        )
        self.k2r_idx_db_file = Path(db_pardir, "key_to_row_indices.db").as_posix()
        self.k2r_idx_db_hash_file = Path(db_pardir, "key_to_row_indices.hash")
        self.memmapped_key_to_row_idx = None

        if storage == "in_memory":
            with open(self.k2r_idx_pkl_file, "rb") as f:
                self.key_to_row_idx = pkl.load(f)
        elif storage == "memory_mapped":
            self.memmapped_key_to_row_idx = lmdb_key_to_row_indices_from_dict(
                db_file=self.k2r_idx_db_file,
                db_hash_file=self.k2r_idx_db_hash_file,
                k2r_idx_file=self.k2r_idx_pkl_file,
            )
            self.key_to_row_idx = self.memmapped_key_to_row_idx
        else:
            raise ValueError

    def close(self):
        """Close LMDB environment."""
        if self.memmapped_key_to_row_idx is not None:
            self.memmapped_key_to_row_idx.close()

    def __enter__(self):
        """Act as a context manager."""
        return self

    def __exit__(self, *args, **kwargs):
        """Cleanups before exiting the context."""
        self.close()

    def set_storage_type(self, storage: str) -> None:
        """Select if the mapping should be kept in memory or saved on disk and memory mapped using
        lmdb.

        Args:
            storage: where to save the data. Acceptable options are ``memory_mapped`` and ``in_memory``.
        """
        if storage == "memory_mapped":
            if self.memmapped_key_to_row_idx is None:
                self.memmapped_key_to_row_idx = lmdb_key_to_row_indices_from_dict(
                    db_file=self.k2r_idx_db_file,
                    db_hash_file=self.k2r_idx_db_hash_file,
                    k2r_idx=self.key_to_row_idx,
                )
            self.key_to_row_idx = self.memmapped_key_to_row_idx
        elif storage == "in_memory":
            if not isinstance(self.key_to_row_idx, dict):
                with open(self.k2r_idx_pkl_file, "rb") as f:
                    self.key_to_row_idx = pkl.load(f)
        else:
            msg = f"Only supported storage formats are 'memory_mapped' and 'in_memory'. Got: '{storage}'"
            raise ValueError(msg)

    def get(self, key: str, default: Optional[Any]) -> Any:
        """Similar to python's ``dict.get`` method."""
        val = self.key_to_row_idx.get(key, None)
        if val is None:
            return default
        else:
            return val

    def keys(self):
        """Similar to python's ``dict.keys``."""
        for key in self.key_to_row_idx.keys():
            yield key

    def values(self):
        """Similar to python's ``dict.values``."""
        for val in self.key_to_row_idx.values():
            yield val

    def items(self):
        """Similar to python's ``dict.items``."""
        for key, val in self.key_to_row_idx.items():
            yield (key, val)

    def __len__(self) -> int:
        """Similar to python's ``len(dict)``."""
        return len(self.key_to_row_idx)

    def __contains__(self, key: str) -> bool:
        """Support python's ``in`` operator like ``key in dict``."""
        return key in self.key_to_row_idx

    def __iter__(self):
        """Support iterating directly over ``KeyToRowIndices`` instances similar to python
        dictionaries."""
        return self.keys()

    def __getitem__(self, key: str) -> List[int]:
        """support indexing operator."""
        return self.key_to_row_idx[key]
