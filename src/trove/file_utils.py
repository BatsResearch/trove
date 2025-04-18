"""Helpers to deal with files and directories on disk.

Utilities in this file should be general and not limited to Trove. I.e., they should not contain
logic of tasks, etc. They are just convenience tools.
"""

import functools
import json
import os
import shutil
import stat
import weakref
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
from uuid import uuid4

import huggingface_hub.utils as hub_utils
import xxhash
from datasets.utils._filelock import FileLock
from tqdm import tqdm
from transformers.utils import logging as hf_logging

from . import cache_manager, config
from .logging_utils import get_logger_with_config, rpath

logger, logging_conf = get_logger_with_config("trove")

hf_logger = hf_logging.get_logger(__name__)


class _TempPath:
    """A class representing a path that should be removed before exit.

    Adapted from `here <https://github.com/huggingface/datasets/blob/17f17b3fe7f276e1b019cca8aa651bf7c818a928/src/datasets/fingerprint.py#L45>`_ .

    You should **NOT** use instances of this class. You should use ``_TempPath.register("path/to/be/deleted")`` or
    ``_TempPath("path/to/be/deleted")`` and forget about it. We just need to create an object to be able
    to use `weakref` module to call cleanup functions.
    """

    # list of paths that are registered for removal
    _registered_paths: List = list()
    # keep a reference of instances of this class to be even more confident that they are
    # not garbage collected before reaching the end of the execution.
    _instances: List = list()

    @classmethod
    def register(cls, path: os.PathLike) -> None:
        """Register a path to be removed before exit if not already registered."""
        realp = realpath(path)
        if realp not in cls._registered_paths:
            cls._instances.append(cls(path))
            cls._registered_paths.append(realp)

    def __init__(self, path: os.PathLike):
        """Register a path to be removed before exit."""
        self.path = path
        # The finalizer is created within the object itself
        # and moreoever the callback function is a bound method of the object
        # itself. So, this object is never garbage collected and the cleanup is only called at
        # interpreter shutdown. This makes sure the directory is not removed while the code is still running.
        self._finalizer = weakref.finalize(self, self._cleanup)

    def _cleanup(self):
        """Remove the path if possible."""
        logger.debug(f"Remove path: {rpath(self.path)}")
        safe_remove_path(self.path)

    def cleanup(self):
        if self._finalizer.detach():
            self._cleanup()


def register_for_removal(path: os.PathLike):
    """Register path to be deleted before exit.

    .. warning::

        In distributed environments, call this function ONLY from one process and
        that should be the process that lasts the LONGEST. If the process that calls
        this function terminates before other processes, it will delete the registered path
        and if other running processes try to access it, they will either raise an exception or
        even worse, generate logically wrong results, which is hard to debug.

    Args:
        path: path to delete before the process exits.
    """
    _TempPath.register(path)


def write_qrels(
    qrels: Dict[str, Dict[str, Union[int, float]]],
    filepath: os.PathLike,
    format: str = "tsv",
) -> None:
    """Writes qrels to a file.

    Args:
        qrels: qrels in nested dict format. ``qrels[qid][docid]`` is the score (or annotation) between `qid` and `docid`.
        filepath: path to target file.
        format: Format of the target file. There are two options:
            * ``tsv`` is the standard qrel format in a tsv file with three columns ("query-id", "corpus-id", "score")
            * ``grouped`` is a json lines file where each record has three keys. ``qid`` is the ID of the current query.
              ``docid`` and ``score`` are two lists of the same size that hold the ID and score of the related documents for ``qid``
    """
    if format not in ["tsv", "grouped"]:
        msg = f"Possible formats for writing qrel records are 'tsv' and 'grouped'. Got: '{format}'"
        raise ValueError(msg)

    if format == "tsv":
        headers = ["query-id", "corpus-id", "score"]
        with CSVWriter(path=filepath, headers=headers, sep="\t") as writer:
            for qid, topdocs in qrels.items():
                for docid, score in topdocs.items():
                    writer.add_one((qid, docid, score))
    elif format == "grouped":
        with JSONLinesWriter(path=filepath) as writer:
            for qid, docdata in qrels.items():
                rec = {
                    "qid": qid,
                    "docid": list(docdata.keys()),
                    "score": list(docdata.values()),
                }
                writer.add_one(rec)


class CSVWriter:
    def __init__(
        self,
        path: os.PathLike,
        headers: Optional[List[str]] = None,
        sep: str = ",",
        chunk_size: Optional[int] = None,
    ) -> None:
        """Open a CSV file for writing.

        Example::

            with CSVWriter('/path/to/file.csv', headers=['qid', 'docid', 'score'], sep=',') as writer:
                writer.add([('a', 'b', 1), ('b', 'c', 2)])
                writer.add_one(('e', 'f', 3))

        Args:
            path: file to open.
            headers: a list of column names. If ``None``, file starts with the first row of data.
            sep: delimiter character
            chunk_size: flush the buffer every ``chunk_size`` records.
        """
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        self.fp = open(path, "w")
        self.sep = sep

        if headers is not None:
            self.fp.write(self.sep.join(headers) + "\n")

        if chunk_size is None:
            self.chunk_size = 1_000
        else:
            self.chunk_size = chunk_size
        self.line_buffer = list()

    def __enter__(self):
        """Act as a context manager."""
        return self

    def __exit__(self, *args, **kwargs):
        """Cleanups before exiting the context."""
        self.close()

    @staticmethod
    def write_to_file(
        records: Iterable[Any],
        path: os.PathLike,
        headers: Optional[List[str]] = None,
        sep: str = ",",
    ) -> None:
        """Write tuples to csv file.

        This methods opens the file, writes the records and closes the file.
        So, there is no incremental writing. It is mostly a convenience method for using
        ``CSVWriter`` when you have all the records (or a generator of records) ready before attempting to write them.

        Args:
            records: List/iterable of tuples to write to file.
            path: file to open.
            headers: a list of column names. If ``None``, file starts with the first row of data.
            sep: delimiter character
        """
        with CSVWriter(path=path, headers=headers, sep=sep) as writer:
            writer.add(records)

    def add(self, items: Iterable[Any]) -> None:
        """Write a list of tuples to file."""
        for item in items:
            self.line_buffer.append(self.sep.join([str(i) for i in item]))
            if len(self.line_buffer) >= self.chunk_size:
                self.flush()

    def add_one(self, item: Any) -> None:
        """Write one object to file."""
        self.add([item])

    def flush(self) -> None:
        """Flush the content of the buffer to file."""
        if len(self.line_buffer) != 0:
            self.fp.write("\n".join(self.line_buffer) + "\n")
            self.line_buffer = list()

    def close(self) -> None:
        """Flush the buffer and close the file."""
        self.flush()
        self.fp.close()


class JSONLinesWriter:
    def __init__(
        self, path: os.PathLike, chunk_size: Optional[int] = None, **kwargs
    ) -> None:
        """Open a json lines file for writing.

        Example::

            with JSONLinesWriter('/path/to/file.jsonl') as writer:
                writer.add([{'a': 1}, {'b': 2}])
                writer.add_one({'c': 3})

        Args:
            path: file to open.
            chunk_size: flush the buffer every ``chunk_size`` records.
            **kwargs: keyword arguments passed to ``json.dumps()``
        """
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        self.fp = open(path, "w")
        self.json_kw = kwargs
        if chunk_size is None:
            self.chunk_size = 1_000
        else:
            self.chunk_size = chunk_size
        self.line_buffer = list()

    def __enter__(self):
        """Act as a context manager."""
        return self

    def __exit__(self, *args, **kwargs):
        """Cleanups before exiting the context."""
        self.close()

    @staticmethod
    def write_to_file(records: Iterable[Any], path: os.PathLike, **kwargs) -> None:
        """Write records to json lines file.

        This methods opens the file, writes the records and closes the file.
        So, there is no incremental writing. It is mostly a convenience method for using
        ``JSONLinesWriter`` when you have all the records (or a generator of records) ready before attempting to write them.

        Args:
            records: List/iterable of objects to write to file.
            path: json lines file to write records to.
            **kwargs: keyword arguments passed to ``json.dumps()``
        """
        with JSONLinesWriter(path=path, **kwargs) as writer:
            writer.add(records)

    def add(self, items: Iterable[Any]) -> None:
        """Write a list of objects to file."""
        for item in items:
            self.line_buffer.append(json.dumps(item, **self.json_kw))
            if len(self.line_buffer) >= self.chunk_size:
                self.flush()

    def add_one(self, item: Any) -> None:
        """Write one object to file."""
        self.add([item])

    def flush(self) -> None:
        """Flush the content of the buffer to file."""
        if len(self.line_buffer) != 0:
            self.fp.write("\n".join(self.line_buffer) + "\n")
            self.line_buffer = list()

    def close(self) -> None:
        """Flush the buffer and close the file."""
        self.flush()
        self.fp.close()


def easyfilelock(path: os.PathLike, *args, **kwargs) -> FileLock:
    """Indirectly locks a file for writing.

    Instead of locking the main file, it creates a new lock file by
    adding ``.lock`` suffix to the main filepath and locks this new file instead.

    See why: https://py-filelock.readthedocs.io/en/latest/index.html

    Args:
        path (os.PathLike): filepath to lock for writing

    Returns:
        an instance of ``FileLock`` for writing to ``path``.
    """
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    lock_file = Path(path).as_posix() + ".lock"
    return FileLock(lock_file, *args, **kwargs)


def realpath(path: os.PathLike) -> os.PathLike:
    """Return resolved absolute path.

    It is similar to linux ``realpath`` command. If ``path`` does exist, join realpath of its
    parent with its basename.
    """
    path = Path(path)
    if path.exists():
        return path.absolute().resolve().as_posix()
    else:
        basename = path.name
        dirname = path.parent.as_posix()
        return Path(realpath(dirname), basename).as_posix()


def _contains_symlink(path: os.PathLike) -> bool:
    """Check if a path is symlink or contains a symlink.

    Returns:
        True if the path itself is a symlink or it contains a symlink. False otherwise.
    """
    path = Path(path)
    if path.is_symlink():
        return True
    if path.is_dir():
        for subpath in path.iterdir():
            if _contains_symlink(subpath):
                return True
    # If we get to this point, it means the path is not a symlink and we didn't find a symlink
    # in any of its subdirectories. So return False
    return False


def safe_remove_path(path: os.PathLike, try_chmod: bool = False) -> None:
    """Try to remove a directory. Do not raise exceptions on failure.

    Adapted from `here <https://github.com/huggingface/huggingface_hub/blob/26c2d89f5e521b50916fbaf190e5a42705bcb775/src/huggingface_hub/utils/_fixes.py#L43>`_ .

    Args:
        path (os.PathLike): Path to delete.
        try_chmod (bool): When True, if failed to remove ``path``, try to change its permissions and try again.
    """
    _realpath = realpath(path)
    if not Path(_realpath).exists():
        # Nothing to remove
        return
    try:
        if Path(_realpath).is_dir():
            shutil.rmtree(_realpath)
        else:
            Path(_realpath).unlink()
    except:
        if try_chmod:
            # If failed, change permission and try again.
            try:
                os.chmod(_realpath, stat.S_IWRITE)
                if Path(_realpath).is_dir():
                    shutil.rmtree(_realpath)
                else:
                    Path(_realpath).unlink()
            except:
                pass


def call_cleanup(cleanup_fn: str):
    """A decorator to call a clean up function on class methods.

    It ensures to call ``cleanup_fn`` method of the object after the decorated
    function returns or raises an exception.

    Use like this::

        Class A:
            def mycleanup(self):
                ...

            @call_cleanup('mycleanup')
            def some_fn(self, ...):
                ...

    Args:
        cleanup_fn (str): The name of the class method to call.
    """

    def _clean_up_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            obj = args[0]
            try:
                return func(*args, **kwargs)
            finally:
                getattr(obj, cleanup_fn)()

        return wrapper

    return _clean_up_decorator


def hash_file_bytes_functional(
    path: os.PathLike, chunk_size: Optional[int] = None
) -> str:
    """Calculates a non-cryptographic hash of the file content.

    It repeats the calculation every time that it's called.
    Use :func:`hash_file_bytes` that relies on cached results to avoid duplicate computations.

    Args:
        path (os.PathLike): file to calculate the bash for
        chunk_size (Optional[int]): read file in chunks of ``chunk_size`` bytes. The default is 1MB.

    Returns:
        hash of the file content.
    """
    logger.debug("Hashing content of file.")
    if chunk_size is None:
        chunk_size = 2**20
    file_hash = xxhash.xxh64()
    num_bytes = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            num_bytes += len(chunk)
            file_hash.update(chunk)

    file_hash = file_hash.hexdigest()

    formatted_size = tqdm.format_sizeof(num_bytes, divisor=1024, suffix="B")
    logger.debug(f"Calculated hash for file with size: {formatted_size}")
    return file_hash


def hash_file_bytes(path: os.PathLike, chunk_size: Optional[int] = None) -> str:
    """Returns a non-cryptographic hash of the file content.

    It tries to read the hash from cache if it exists. If missing, calculate the hash and
    also write it to a cache file.

    Args:
        path (os.PathLike): file to calculate the hash for.
        chunk_size (Optional[int]): see :func:`hash_file_bytes_functional` for details.

    Returns:
        hash of the file content.
    """
    cache_file = cache_manager.get_cache_dir(
        input_data=path,
        artifact_type="final",
        artifact_content="content_hash",
        fingerprint_src="path",
    )
    cache_file = cache_file.joinpath("unique_hash_of_bytes.txt")
    with easyfilelock(cache_file.as_posix()):
        if not cache_file.exists():
            file_hash = hash_file_bytes_functional(path=path, chunk_size=chunk_size)
            with open(cache_file, "w") as f:
                f.write(file_hash)

    with open(cache_file, "r") as f:
        file_hash = f.read().strip()
    return file_hash


# from here:
# https://github.com/huggingface/huggingface_hub/blob/d30853b36029bd838500614cdae63b3bb55a8950/src/huggingface_hub/file_download.py#L1564
def _chmod_and_move(src: Path, dst: Path) -> None:
    """Set correct permission before moving a blob from tmp directory to cache dir.

    Do not take into account the ``umask`` from the process as there is no convenient way
    to get it that is thread-safe.

    See:
        - `About umask <https://docs.python.org/3/library/os.html#os.umask>`_
        - `Thread-safety <https://stackoverflow.com/a/70343066>`_
        - `About solution <https://github.com/huggingface/huggingface_hub/pull/1220#issuecomment-1326211591>`_
        - `Fix issue <https://github.com/huggingface/huggingface_hub/issues/1141>`_
        - `Fix issue <https://github.com/huggingface/huggingface_hub/issues/1215>`_
    """
    # Get umask by creating a temporary file in the cached repo folder.
    tmp_file = dst.parent.parent / f"tmp_{uuid4()}"
    try:
        tmp_file.touch()
        cache_dir_mode = Path(tmp_file).stat().st_mode
        os.chmod(str(src), stat.S_IMODE(cache_dir_mode))
    except OSError as e:
        hf_logger.warning(
            f"Could not set the permissions on the file '{src}'. "
            f"Error: {e}.\nContinuing without setting permissions."
        )
    finally:
        try:
            tmp_file.unlink()
        except OSError:
            # fails if `tmp_file.touch()` failed => do nothing
            # See https://github.com/huggingface/huggingface_hub/issues/2359
            pass

    shutil.move(str(src), str(dst), copy_function=_copy_no_matter_what)


# from here:
# https://github.com/huggingface/huggingface_hub/blob/d30853b36029bd838500614cdae63b3bb55a8950/src/huggingface_hub/file_download.py#L1599
def _copy_no_matter_what(src: str, dst: str) -> None:
    """Copy file from src to dst.

    If ``shutil.copy2`` fails, fallback to ``shutil.copyfile``.
    """
    try:
        # Copy file with metadata and permission
        # Can fail e.g. if dst is an S3 mount
        shutil.copy2(src, dst)
    except OSError:
        # Copy only file content
        shutil.copyfile(src, dst)


@contextmanager
def atomic_write(file: os.PathLike, root: str = "parent"):
    """Write to file only if the write operation is completed successfully.

    It returns a path to a temporary file that you should use instead of the original file path.
    If the write operation is completed successfully, it moves the temporary file to the original filepath.
    If write operation fails for any reason, it just removes the temporary file if it can (it does not raise errors if
    it fails to remove the temporary file).

    You should always work with the original filepath and only call this context manager
    at the very last moment before opening the file for writing. For example, if you need to check
    for patterns in the filepath or check the file extension, you should use the original filepath.

    Example::

        path = '/some/path/to/my/file.jsonl'
        with atomic_write(path) as new_path:
            with open(new_path, 'w') as f:
                f.write('some content')

    Keep in mind that the temporary filepath is totally random and changes everytime you call this function even with the same arguments.

    .. attention::

        If you are using ``trove.file_utils.easyfilelock()``, you must lock the original filepath and not the one returned by this function.

    Args:
        file: filepath to write the results to if the write operation is successful.
        root: where to create the temporary file. You can avoid copying the temporary file
            and just renaming it (which is much faster for large files) if you choose a
            directory that is on the same filesystem as the original filepath. Valid options are:
                - ``system``: use the system's temporary directory (i.e., wherever python's tempfile module creates its files).
                - ``cache``: use the central cache directory used in trove library
                - ``parent``: create a directory with specific name (name is the value
                    of ``trove.config.TROVE_INCOMPLETE_FILE_DIRNAME`` variable) in the parent directory of filepath.

    Yields:
        the temporary filepath that the user must use for write operations in this context manager.
    """
    valid_roots = ["system", "parent", "cache"]
    if root not in valid_roots:
        msg = f"Valid options for 'root' are '{valid_roots}'. Got: 'root': '{root}'"
        raise ValueError(msg)

    if root == "system":
        temp_dir_cm = hub_utils.SoftTemporaryDirectory()
    else:
        if root == "parent":
            # temp file will be in a subdir in file's parent
            _tr = Path(file).parent.joinpath(config.TROVE_INCOMPLETE_FILE_DIRNAME)
        elif root == "cache":
            # temp file will be in trove's centralized cache
            _tr = cache_manager.get_cache_pardir(
                artifact_content=config.TROVE_INCOMPLETE_FILE_DIRNAME,
                artifact_type="temp",
            )
        else:
            raise ValueError
        temp_dir_cm = nullcontext(Path(_tr).as_posix())

    with temp_dir_cm as temp_dir:
        # Make sure temp files are unique. Although not guaranteed in future versions, try to keep
        # the same base name as the original file so not to break libraries that rely on the extension of  file.
        temp_file = Path(temp_dir, uuid4().hex, Path(file).name).as_posix()
        # create the parent directory to avoid errors
        Path(temp_file).parent.mkdir(exist_ok=True, parents=True)
        try:
            logger.debug(
                f"Start write transaction. Original file: {rpath(file)} Temp file: {rpath(temp_file)}"
            )
            # make it a drop-in replacement
            if isinstance(file, Path):
                yield Path(temp_file)
            else:
                yield temp_file
            # If the caller has not created the temp_file, there is nothing to do.
            if Path(temp_file).exists():
                logger.debug(
                    f"Finish write transaction by moving temp file to actual destination of: {rpath(file)}"
                )
                # Suppress warning from Path.touch() by creating its parent directories
                Path(file).parent.mkdir(exist_ok=True, parents=True)
                # Move the temp file and then remove it
                _chmod_and_move(src=Path(temp_file), dst=Path(file))
        finally:
            logger.debug(
                f"After write transaction. Remove the temp file before exit: {rpath(temp_file)}"
            )
            # The temp file and auxiliary files should be removed whether we succeed or fail
            for _file in Path(temp_file).parent.iterdir():
                Path(_file).unlink(missing_ok=True)
            # Do not make a giant directory with a bunch of empty subdirectories
            # We can delete the dir safely since it is unique to this specific function call
            assert Path(temp_file).parent.exists() and not len(
                list(Path(temp_file).parent.iterdir())
            )
            Path(temp_file).parent.rmdir()


def create_file_name_to_id_mapping(
    files: Union[os.PathLike, List[os.PathLike]], id_length: str = "shortest"
) -> Dict[str, str]:
    """Create unique IDs for a list of files.

    The resulting file ID is a substring of the hexdigest of the hash of the binary content of the file.

    **NOTE: It will fail if a subset of files in ``files`` are copy of each other, i.e., have same hash.**

    Args:
        files (Union[os.PathLike, List[os.PathLike]]): a list of filepath to creates unique IDs for.
        id_length (str): The number of characters in the final ID. Acceptable values are

            * ``shortest``: Start from length=4 and increment the length by 1 until you get unique IDs for all files.
            * ``full``: use the full hash as ID.

    Returns:
        A mapping where keys are the elements of the ``files`` argument and the values are the corresponding unique file IDs.
    """
    if not isinstance(files, list):
        files = [files]

    # Ensure files are of 'str' dtype
    orig_files = [Path(p).as_posix() for p in files]
    realpath_files = [realpath(p) for p in orig_files]
    # the subset of unique files with their actual path (not symlinks, etc.)
    realpath_files = list(set(realpath_files))

    # Try to preserve the original IDs calculated by MaterializedQRel instances
    realpath_to_hash = dict()
    for _realpath in realpath_files:
        realpath_to_hash[_realpath] = hash_file_bytes(_realpath)

    if id_length == "shortest":
        _id_length = 4
    elif id_length == "full":
        _id_length = len(next(iter(realpath_to_hash.values())))
    else:
        msg = f"Possible values for 'id_length' are: 'shortest', 'full'. Got: '{id_length}'"
        raise ValueError(msg)

    while True:
        if _id_length > len(next(iter(realpath_to_hash.values()))):
            msg = "At least two of the files are identical copies of each other and it is not possible to make different IDs for them."
            raise RuntimeError(msg)

        realpath_to_id = {k: v[:_id_length] for k, v in realpath_to_hash.items()}
        if len(set(list(realpath_to_id.values()))) == len(realpath_files):
            break
        else:
            _id_length += 1

    # This block is never executed. It is remained here from previous versions.
    if len(set(list(realpath_to_id.values()))) != len(realpath_files):
        # If hashes are not unique, use sequential IDs
        realpath_to_id = {f: str(i) for i, f in enumerate(list(sorted(realpath_files)))}

    # Convert the mapping keys from resolved absolute path
    # to the exact path that is used by MaterializedQRel instances
    orig_name_to_id = {p: realpath_to_id[realpath(p)] for p in orig_files}
    return orig_name_to_id
