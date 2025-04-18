import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pyarrow as pa
import torch

from .. import config, file_utils
from ..containers.key_to_row_indices import KeyToRowIndices
from ..logging_utils import get_logger_with_config, rpath

logger, logging_conf = get_logger_with_config("trove")


class VectorCacheMixin:
    def __init__(self, cache_file_name: Optional[os.PathLike] = None) -> None:
        """Mixin class for reading and writing dense vectors to pyarrow tables.

        This class provides the functionality to read and write dense vectors to pyarrow tables.
        It is intended to be used as a Mixin that provides this functionality to other classes with
        minimal effort.
        We use it to read and write query and passage embeddings to a cache file during
        evaluation of IR models.

        It only supports reading and writing ``(_id, value)`` tuples.

            * ``_id`` should be of type ``str``
            * ``value`` is a 1D numpy array of type `float32`
            * ``_id`` must uniquely identify each record in the entire dataset

        Args:
            cache_file_name (Optional[os.PathLike]): Path to cache file.
                Cache is disabled if not provided. It is also possible to add it after initialization.
        """
        # pyarrow schema representing a table with two columns,
        # '_id' (of type str) and 'value' (a 1D float vector)
        self.schema = pa.schema(
            [
                pa.field("_id", pa.string()),
                pa.field("value", pa.list_(pa.float32())),
            ]
        )

        self.cache_file_name: Optional[os.PathLike] = None
        # If true, we are allowed to write to cache file,
        # only allowed if cache file does not exist
        self.is_writable: Optional[bool] = None

        # pyarrow table holding cached values
        self.cache_table: Optional[pa.Table] = None
        # A mapping from _id to the corresponding row index in self.cache_table
        self._id_to_cache_idx: Optional[Union[Dict[str, int], KeyToRowIndices]] = None
        # Pyarrow file object used for writing
        self.file_pointer: Optional[pa.NativeFile] = None
        # pyarrow writer
        self.writer: Optional[pa.RecordBatchFileWriter] = None
        # Rows are written in batches. This holds rows that are waiting to be written to file.
        self.buffered_rows: List[Dict] = list()
        # is_writing is True if the cache file is open for writing
        self.is_writing: Optional[bool] = None

        # If not None, save the cache files in a subdir with this name in the directory where
        # cache files would've been saved otherwise.
        # Use this to save/load cache files after some permanent change has been made to the corresponding dataset.
        # This is not cleared when calling self.reset_state. It is permanent.
        # For example, if you do a _hard_ shard on the dataset, you should update this (by calling self.update_cache_subdir)
        # attribute to save the cache for hard sharded dataset in a separate subdirectory
        self.cache_variant_subdir: Optional[os.PathLike] = None

        self.update_cache_file_name(file_name=cache_file_name, load=False)

    @property
    def is_cache_available(self) -> bool:
        return self.cache_table is not None

    @property
    def effective_cache_file_name(self) -> Optional[os.PathLike]:
        """Effective cache file name after taking into account the ``cache_variant_subdir``
        property.

        You should always use this property to read/write cache files.
        """
        if self.cache_file_name is None:
            return None
        if self.cache_variant_subdir is None:
            return self.cache_file_name

        _fn = Path(self.cache_file_name)
        fname = _fn.parent.joinpath(self.cache_variant_subdir, _fn.name).as_posix()
        return fname

    def reset_state(self) -> None:
        """Resets the internal state of the class like a fresh instance without a cache filename.

        Does NOT change files on disk.
        """
        # Not allowed while writing to avoid corrupted cache files.
        assert (
            not self.is_writing
        ), "Attempting to reset the state while writing to the cache"

        self.cache_file_name = None
        self.is_writable = None
        self.cache_table = None
        if hasattr(self._id_to_cache_idx, "close"):
            self._id_to_cache_idx.close()
        self._id_to_cache_idx = None
        self.file_pointer = None
        self.writer = None
        self.buffered_rows = list()
        self.is_writing = None

    def unload_cache(self) -> None:
        """Purge the cache and index lookup tables that are loaded into memory.

        It reverses the impact of :meth:`load_cache`
        """
        # This will never be triggered because if the cache is loaded, it is not writable
        # But do it for consistency to emphasize for developers that no other operations
        # are allowed while writing to cache
        assert not self.is_writing, "Attempting to unload cache while writing"
        self.cache_table = None
        if hasattr(self._id_to_cache_idx, "close"):
            self._id_to_cache_idx.close()
        self._id_to_cache_idx = None

    def update_cache_subdir(
        self, subdir: Optional[os.PathLike], append: bool = True, load: bool = True
    ) -> None:
        """Update the cache nested subdir (e.g., after some permanent change to the corresponding
        dataset).

        When some permanent change has been made to the corresponding dataset (e.g., hard sharding),
        you should to let the ``VectorCache`` know where to save/load the cache files for the new dataset.
        This dataset also reset the state of loaded cache files, etc. (But does not change files on disk.).

        After updating cache subdir, it calls to :meth:`update_cache_file_name` initialize the new cache
        if it exists.

        Args:
            subdir (Optional[os.PathLike]): New cache files are saved in a subdirectory with this name
                in the parent directory that would've contained the original cache files.
            append (bool): overwrite or append to existing cache ``subdir`` if it exists.
            load (bool): If true, load the new cache file if it exists (the state of the previous cache tables
                is always cleared even if load is set to ``False``).
        """
        if append and subdir is None:
            msg = "cannot append `None` to existing `self.cache_variant_subdir`."
            raise TypeError(msg)

        # Keep track of the main cache filename
        # We are only changing the variant and not the main cache file name.
        prev_filename = self.cache_file_name
        if prev_filename is not None:
            prev_filename = Path(prev_filename).as_posix()

        if append:
            if self.cache_variant_subdir is None:
                self.cache_variant_subdir = Path(subdir).as_posix()
            else:
                self.cache_variant_subdir = Path(
                    self.cache_variant_subdir, subdir
                ).as_posix()
        else:
            msg = "Overwriting 'self.cache_variant_subdir' is not yet supported."
            raise NotImplementedError(msg)

        self.update_cache_file_name(file_name=prev_filename, load=load)

    def _get_vector_cache_shard_files(self) -> List[str]:
        """Returns a list of paths to cache shard files."""
        _cache_shard_files = list()
        if Path(self.effective_cache_file_name).parent.exists():
            # cache files can sometimes be sharded like 'file-000-of-001.arrow'
            # also exclude temporary incomplete files that were not removed
            _cache_shard_files = [
                p.absolute().as_posix()
                for p in Path(self.effective_cache_file_name).parent.rglob("*")
                if not p.is_dir()
                and p.name.startswith(Path(self.effective_cache_file_name).stem)
                and config.TROVE_INCOMPLETE_FILE_DIRNAME not in p.parts
            ]
        return sorted(_cache_shard_files)

    def update_cache_file_name(
        self, file_name: Optional[os.PathLike] = None, load: bool = True
    ) -> None:
        """Resets the state of the class and points to the new cache file.

        Args:
            file_name (Optional[os.PathLike]): New cache file.
            load (bool): If true and ``file_name`` is not ``None``, load the new cache file.
        """
        # Clear the old cache first
        self.reset_state()
        self.cache_file_name = file_name
        if self.effective_cache_file_name is None:
            # Nothing to do
            return

        _cache_shard_files = self._get_vector_cache_shard_files()
        # We are only allowed to write to this file
        # if it does not exist (and none of its sharded variants exist)
        self.is_writable = len(_cache_shard_files) == 0
        if load:
            self.load_cache()

    def load_cache(self) -> None:
        """Load the cache file as a memory mapped Arrow table."""
        # Writing should be finished (or not started at all) before loading the cache
        assert not self.is_writing, "Attempting the load the cache while writing"

        if self.cache_table is not None:
            # It is already loaded
            return

        if (
            self.effective_cache_file_name is None
            or not Path(self.effective_cache_file_name).parent.exists()
        ):
            # there is nothing to load
            return

        _cache_shard_files = self._get_vector_cache_shard_files()
        if not len(_cache_shard_files):
            # There is nothing to load
            return

        logger.info(
            f"Loading vector cache pyarrow table from {rpath(self.effective_cache_file_name)}. Num shards: {len(_cache_shard_files)}"
        )

        # load the cache files as memory mapped arrow tables
        cache_table = list()
        for file in _cache_shard_files:
            with pa.memory_map(file, "rb") as src:
                with pa.ipc.open_file(src) as reader:
                    cache_table.append(reader.read_all())

        if len(cache_table) == 1:
            cache_table = cache_table[0]
        else:
            cache_table = pa.concat_tables(cache_table)

        cached_ids = cache_table["_id"].to_numpy().tolist()
        assert len(cached_ids) == len(
            set(cached_ids)
        ), "Found duplicate record IDs in the cache"

        assert set(self.all_rec_ids) == set(
            cached_ids
        ), "Cached IDs are different from dataset IDs"

        logger.debug("Create cache id to row index.")
        # self._id_to_cache_idx = KeyToRowIndices(
        # dataset=datasets.Dataset.from_dict({"_id": cached_ids}),
        #     id_field="_id",
        #     num_proc=8,
        # )
        self._id_to_cache_idx = {_id: [_idx] for _idx, _id in enumerate(cached_ids)}
        self.cache_table = cache_table

    def get_cached_value(self, _id: str) -> Optional[np.ndarray]:
        """If possible, loads the cached value for the given ``_id``

        .. warning::

            The returned cached value shares memory with the arrow table and should not be modified in place.
            **Do not change this array in place.**

        Args:
            _id (str): Load the cached value corresponding to this unique ``_id``

        Returns:
            If cache exists, return the cached value as numpy array, otherwise, return ``None``.
        """
        if self.cache_table is None or self.is_writing:
            # You cannot read from cache while writing or if cache does not exist
            return None
        idx = self._id_to_cache_idx[_id][0]
        value = self.cache_table["value"][idx].values.to_numpy()
        return value

    @contextmanager
    def open_cache_io_streams(self):
        """A context manager to prepare for writing to cache files.

        It opens the cache file and creates the necessary write handlers.
        Should be used like::

            with instance.open_cache_io_streams():
                instance.cache_records(...) # You write to cache here
        """
        if self.effective_cache_file_name is None or not self.is_writable:
            # It is not possible to write to cache.
            # Return a dummy context manager to keep the user's code consistent whether they write to cache or not
            # We will raise an exception in other methods if the user tries to add records to cache when not possible
            yield
        else:
            assert (
                not self.is_writing
            ), "Attempting to write to file that is already open"
            # The shard information should be set by the class that inherits from this Mixin
            if (
                self.shard_idx is not None
                and self.shard_idx >= 0
                and self.num_shards is not None
                and self.num_shards > 0
            ):
                # If sharded creates files like '/path/to/cache/dir/cache_file-000-001.arrow'
                _pardir = Path(self.effective_cache_file_name).parent
                _cache_stem = Path(self.effective_cache_file_name).stem
                width = len(str(int(self.num_shards))) + 1
                suffix = f"-{self.shard_idx:0{width}}-of-{self.num_shards:0{width}}"
                sink_file_name = Path(_pardir).joinpath(
                    _cache_stem + suffix + Path(self.effective_cache_file_name).suffix
                )
            else:
                sink_file_name = Path(self.effective_cache_file_name)

            assert (
                not sink_file_name.exists()
            ), f"Cache file already exists:\n{sink_file_name.as_posix()}"
            sink_file_name.parent.mkdir(exist_ok=True, parents=True)

            logger.debug(
                f"Opened cache file for writing: {rpath(sink_file_name.as_posix())}"
            )
            with file_utils.atomic_write(file=sink_file_name, root="parent") as tfile:
                self.file_pointer = pa.OSFile(tfile.as_posix(), "wb")
                self.writer = pa.ipc.new_file(self.file_pointer, self.schema)
                self.is_writing = True
                # You can only write to a cache file once. No updating allowed.
                self.is_writable = False
                try:
                    yield
                finally:
                    # Write all the buffered records to file before closing the streams
                    self.flush()
                    self.writer.close()
                    self.file_pointer.close()
                    self.is_writing = False

    def cache_records(
        self,
        rec_id: List[str],
        value: Union[torch.Tensor, np.ndarray, List[np.ndarray]],
        chunk_size: int = 1_000,
    ) -> None:
        """Add a batch of records to the cache.

        The write is not immediate.
        It is written in chunks once a sufficient number of records are buffered.

        Args:
            rec_id (List[str]): a list of unique ``_id`` values
            value (Union[torch.Tensor, np.ndarray, List[np.ndarray]]): values to cache.
                It should either be a 2D ``torch.Tensor`` or 2D ``np.ndarray``.
                Each row is treated as a record to be written to the cache.
            chunk_size (int): The number of records to write at a time.
        """
        if self.effective_cache_file_name is None:
            # Do not raise exception if we are using the dataset without caching
            return

        # caching is enabled. Make sure we can write to cache file
        assert (
            self.is_writing
            and self.writer is not None
            and self.file_pointer is not None
            and self.file_pointer is not None
            and not self.file_pointer.closed
        )

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()

        if isinstance(value, np.ndarray):
            # Convert 2D array to a list of 1D arrays
            value = np.vsplit(value, value.shape[0])

        # Add to buffer and write to file later
        for _id, v in zip(rec_id, value):
            self.buffered_rows.append(
                {"_id": _id, "value": v.flatten().astype(np.float32)}
            )

        if len(self.buffered_rows) >= chunk_size:
            self.flush(chunk_size=chunk_size)

    def flush(self, chunk_size: int = 1_000) -> None:
        """Write buffered records to file.

        Args:
            chunk_size (int): The size of chunked arrays to write to arrow file.
        """
        # Make sure it is possible to write to cache file
        assert (
            self.is_writing
            and self.writer is not None
            and self.file_pointer is not None
            and not self.file_pointer.closed
            and self.file_pointer is not None
            and not self.file_pointer.closed
        )
        if len(self.buffered_rows):
            table = pa.Table.from_pylist(self.buffered_rows, schema=self.schema)
            self.writer.write_table(table, chunk_size)
            self.buffered_rows = []

    def _shard_checks_by_cache(self) -> None:
        """Do nothing if it is ok to shard the dataset, otherwise raise an exception."""
        assert (
            not self.is_writing
        ), "Attempting to shard the data while writing to cache file"

    @property
    def all_rec_ids(self) -> List[str]:
        """Returns the list of all record ids in the dataset across all shards."""
        raise NotImplementedError
