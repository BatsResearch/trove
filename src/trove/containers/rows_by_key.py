import itertools
import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, List, Optional, Union

import datasets
from datasets.fingerprint import Hasher

from .. import cache_manager
from ..logging_utils import get_logger_with_config
from .key_to_row_indices import KeyToRowIndices

logger, logging_conf = get_logger_with_config("trove")


class RowsByKeySingleSource:
    def __init__(
        self,
        dataset: Optional[datasets.Dataset] = None,
        filepath: Optional[os.PathLike] = None,
        key_field: Optional[str] = None,
        new_key_field: Optional[str] = None,
        add_filepath_field: bool = False,
        allow_duplicates: bool = True,
        unique_field_subset: Optional[Union[str, List[str]]] = None,
        ignore_main_data: bool = False,
        num_proc: Optional[int] = None,
    ) -> None:
        """A custom collection that retrieves a list of rows that have a specific value in the
        specified column.

        The data source can be a JSONL file or an HF ``datasets.Dataset`` instance.

        Args:
            dataset (Optional[datasets.Dataset]): Use this ``datasets.Dataset`` instance as source
            filepath (Optional[os.PathLike]): Read data from file
            key_field (Optional[str]): name of the field to use for querying rows (e.g., ``'qid'``)
            new_key_field (Optional[str]): If given, change the name of the key field to ``new_key_field``
            add_filepath_field (bool): If true, add a new field to each row that contains the name of the source file
            allow_duplicates (bool): Not implemented
            unique_field_subset (Optional[Union[str, List[str]]]): Not implemented
            ignore_main_data (bool): if true, do not return the actual data fields for each row and just
                return the metadata fields like value of ``'key_field'`` and ``'filepath'`` if it exists.
            num_proc (Optional[int]): Max number of processes when reading and pre-processing the data.
        """
        # You should specify one and only one source of data
        if dataset is not None and filepath is not None:
            msg = f"Only one source of data for rows is supported. Got: 'dataset': '{dataset}' and 'filepath': '{filepath}'"
            raise ValueError(msg)

        if not allow_duplicates or unique_field_subset is not None:
            msg = "Checking for duplicates is not supported yet"
            raise NotImplementedError(msg)

        self.filepath = filepath
        self.add_filepath_field = add_filepath_field
        self.ignore_main_data = ignore_main_data
        # A unique fingerprint for the contents and output of this collection
        # do not use this directly. Instead use 'self.fingerprint' attribute
        self._unique_fingerprint = None

        if dataset is not None:
            self.dataset = dataset
        elif self.filepath is not None:
            if Path(self.filepath).suffix not in [".json", ".jsonl"]:
                msg = "Only json/jsonl files are supported"
                raise NotImplementedError(msg)
            logger.debug(f"Reading file for RowsByKeySingleSource:\n{self.filepath}")
            self.dataset = datasets.load_dataset(
                "json", data_files=filepath, split="train"
            )
        else:
            # There are no records. It is an empty container
            k = "_id" if key_field is None else key_field
            self.dataset = datasets.Dataset.from_dict({k: []})

        if key_field is None:
            # If key_field is not explicitely provided, check if any of the predefined potential id fields are present
            ds_fields = self.dataset.column_names
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
            if len(ds_id_fields) == 1:
                key_field = ds_id_fields[0]
            else:
                if len(ds_id_fields) == 0:
                    msg = "`key_field` is not explicitely specified and none of the predefined potential id fields are present"
                else:
                    msg = f"`key_field` is not explicitely specified and more than one predefined potential ID field found: {ds_id_fields}"
                raise RuntimeError(msg)

        assert key_field is not None and key_field in self.dataset.column_names

        if new_key_field is not None and new_key_field != key_field:
            # Change the name of the key column
            assert new_key_field not in self.dataset.column_names
            self.dataset = self.dataset.rename_columns({key_field: new_key_field})
            key_field = new_key_field

        self.key_field = key_field

        # Create a mapping from values of column 'key_field' to the list of indices of rows that contain those values
        self.key_to_row_idx = KeyToRowIndices(
            dataset=self.dataset, id_field=self.key_field, num_proc=num_proc
        )

        # Add a new field named self.filepath_field that holds the name of the source file.
        self.filepath_field = "_file"
        if self.add_filepath_field and self.filepath_field in self.dataset.column_names:
            msg = (
                "You've asked to add the filename of the source data as a new key (`_file`) to each record."
                "But field `_file` already exists. We do not overwrite existing data."
            )
            raise RuntimeError(msg)

    @property
    def fingerprint(self) -> str:
        """Calculates a unique fingerprint for the contents and output of this container.

        Containers with the same fingerprint generate the same output.
        """
        if self._unique_fingerprint is not None:
            return self._unique_fingerprint
        # The combined hash of the following attributes uniquely identifies this container
        # A container with the same hash always produces the same output
        # Technically, 'self.ignore_main_data' should also be included in the hash since it impacts the output
        # But, because 'self.ignore_main_data' can change multiple times during a single execution, we ignore it for now.
        hasher = Hasher()
        hasher.update(self.filepath)
        hasher.update(self.add_filepath_field)
        hasher.update(cache_manager.hf_dataset_fingerprint(self.dataset)[0])
        hasher.update(self.key_field)
        hasher.update(self.filepath_field)

        self._unique_fingerprint = hasher.hexdigest()
        return self._unique_fingerprint

    def update_fingerprint(self) -> str:
        """Invalidate and recompute the unique fingerprint for this container."""
        self._unique_fingerprint = None
        return self.fingerprint

    def set_index_lookup_storage_type(self, storage: str) -> None:
        """Select if the key to row index lookup table should be stored in memory or in memory-
        mapped lmdb dict."""
        self.key_to_row_idx.set_storage_type(storage=storage)

    @contextmanager
    def meta_info_only(self):
        """A context manager to temporarily ignore the actual data and just return the metadata
        about rows."""
        orig_ignore_main_data = self.ignore_main_data
        self.ignore_main_data = True
        try:
            yield
        finally:
            self.ignore_main_data = orig_ignore_main_data

    def keys(self):
        """similar to ``dict.keys``"""
        return self.key_to_row_idx.keys()

    def __getitem__(self, key: Any) -> List[Any]:
        """Uses precomputed list of indices of all rows that their ``key_field == key`` to extract
        the subset of rows."""
        row_indices = self.key_to_row_idx[key]
        assert len(row_indices), "Key not found"
        if self.ignore_main_data:
            if self.add_filepath_field:
                rows = [
                    {self.key_field: key, self.filepath_field: self.filepath}
                    for _ in range(len(row_indices))
                ]
            else:
                rows = [{self.key_field: key} for _ in range(len(row_indices))]
            return rows
        rows = list()
        for idx in row_indices:
            rec = self.dataset[idx]
            # Double check that since we've created the key_to_idx mapping, the dataset has not changed.
            assert rec[self.key_field] == key
            if self.add_filepath_field:
                rec[self.filepath_field] = self.filepath
            rows.append(rec)
        return rows

    def get(self, key: Any, default: Optional[Any] = None) -> Optional[Any]:
        """similar to ``dict.get``"""
        if key in self.key_to_row_idx and len(self.key_to_row_idx[key]):
            return self[key]
        else:
            return default


class RowsByKey:
    def __init__(
        self,
        record_mapping: Optional[
            Union[RowsByKeySingleSource, List[RowsByKeySingleSource]]
        ] = None,
        dataset: Optional[Union[datasets.Dataset, List[datasets.Dataset]]] = None,
        filepath: Optional[Union[os.PathLike, List[os.PathLike]]] = None,
        key_field: Optional[str] = None,
        new_key_field: Optional[str] = None,
        add_filepath_field: bool = False,
        allow_duplicates_per_id_lazy: bool = True,
        allow_duplicates: bool = True,
        unique_field_subset: Optional[Union[str, List[str]]] = None,
        ignore_main_data: bool = False,
        num_proc: Optional[int] = None,
    ) -> None:
        """It combines the results of multiple
        :class:`~trove.containers.rows_by_key.RowsByKeySingleSource` instances.

        The rest of the arguments are the same as :class:`~trove.containers.rows_by_key.RowsByKeySingleSource`

        Args:
            record_mapping (Optional[Union[RowsByKeySingleSource, List[RowsByKeySingleSource]]]):
                a single or a list of ``RowsByKeySingleSource`` instances to use as data source
            dataset (Optional[Union[datasets.Dataset, List[datasets.Dataset]]]): a single or a list of datasets
            filepath (Optional[Union[os.PathLike, List[os.PathLike]]]): a single or a list of files as data source
            allow_duplicates_per_id_lazy (bool): If False, when calling ``__getitem__``, it raises an exception if there are more than one row for the key
        """
        self.allow_duplicates_per_id_lazy = allow_duplicates_per_id_lazy
        # A unique fingerprint for the contents and output of this collection
        # do not use this directly. Instead use 'self.fingerprint' attribute
        self._unique_fingerprint = None

        if (
            sum([record_mapping is not None, dataset is not None, filepath is not None])
            > 1
        ):
            msg = "You can at most use one of the 'record_mapping', 'dataset', or 'filepath' as data source."
            raise ValueError(msg)

        if record_mapping is not None:
            if not isinstance(record_mapping, list):
                record_mapping = [record_mapping]
            self.record_mappings = record_mapping
        elif dataset is not None or filepath is not None:
            # Creates lists even if there is only one item
            if dataset is not None:
                if not isinstance(dataset, list):
                    dataset = [dataset]
                filepath = [filepath] * len(dataset)
            elif filepath is not None:
                if not isinstance(filepath, list):
                    filepath = [filepath]
                dataset = [dataset] * len(filepath)

            # Create Mappings for each data source
            self.record_mappings = [
                RowsByKeySingleSource(
                    dataset=ds,
                    filepath=path,
                    key_field=key_field,
                    new_key_field=new_key_field,
                    add_filepath_field=add_filepath_field,
                    allow_duplicates=True,
                    unique_field_subset=None,
                    ignore_main_data=ignore_main_data,
                    num_proc=num_proc,
                )
                for ds, path in zip(dataset, filepath)
            ]
        else:
            self.record_mappings = []

        self.check_duplicates(
            allow_duplicates=allow_duplicates, unique_field_subset=unique_field_subset
        )

    @property
    def fingerprint(self) -> str:
        """Calculates a unique fingerprint for the contents and output of this container.

        Containers with the same fingerprint generate the same output.
        """
        # The content and output of this container depends on the content of its record mappings
        # So we use the combined fingerprint of its record mappings as its fingerprint
        if self._unique_fingerprint is not None:
            return self._unique_fingerprint
        hasher = Hasher()
        for rmap in self.record_mappings:
            hasher.update(rmap.fingerprint)

        self._unique_fingerprint = hasher.hexdigest()
        return self._unique_fingerprint

    def update_fingerprint(self) -> str:
        """Invalidate and recompute the unique fingerprint for this container."""
        self._unique_fingerprint = None
        return self.fingerprint

    def check_duplicates(
        self,
        allow_duplicates: bool = True,
        unique_field_subset: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """Check that there are no duplicate rows across all collections."""
        if not allow_duplicates or unique_field_subset is not None:
            raise NotImplementedError

    def set_index_lookup_storage_type(self, storage: str) -> None:
        """Select if the key to row index lookup table should be stored in memory or in memory-
        mapped lmdb dict."""
        for rmap in self.record_mappings:
            rmap.set_index_lookup_storage_type(storage=storage)

    @contextmanager
    def meta_info_only(self):
        """A context manager to temporarily ignore the actual data and just return the metadata
        about rows."""
        with ExitStack() as stack:
            for rmap in self.record_mappings:
                stack.enter_context(rmap.meta_info_only())
            yield

    def keys(self):
        """Merges the keys for all data sources."""
        return itertools.chain.from_iterable(
            [rmap.keys() for rmap in self.record_mappings]
        )

    def get(self, key: Any, default: Optional[Any] = None) -> Optional[Any]:
        """Merges the results returned by each of the data sources."""
        rows = list()
        for rec_map in self.record_mappings:
            rows.extend(rec_map.get(key, []))
        if len(rows):
            assert self.allow_duplicates_per_id_lazy or len(rows) == 1
            return rows
        else:
            return default

    def __getitem__(self, key: Any) -> List[Any]:
        """Same as ``get()`` method but to mimic python ``dict[key]`` interface."""
        rows = self.get(key, None)
        # The key does not exist
        assert rows is not None, "Key not found"
        return rows
