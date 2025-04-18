"""Logic for managing the univeral artifact cache of the package."""

import datetime
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from uuid import uuid4

import datasets
import huggingface_hub as hf_hub
from datasets.fingerprint import Hasher

from . import config, file_utils

# This will be set to 'True' if the process temporary cache pardir is registered to be
# removed before interpreter shutdown
_TEMP_PARDIR_WILL_BE_REMOVED = False


def _register_temp_pardir_for_removal() -> None:
    """Register temporary cache pardir to be removed before interpreter shutdown."""
    global _TEMP_PARDIR_WILL_BE_REMOVED
    if _TEMP_PARDIR_WILL_BE_REMOVED:
        return
    _TEMP_PARDIR_WILL_BE_REMOVED = True
    # Since the temp cache dir is exclusive to each process, we can safely remove
    # it before exit even if there are other processes that are still running (in a distributed environment).
    temp_pardir = Path(get_cache_pardir(uuid4().hex, "temp")).parent.as_posix()
    file_utils.register_for_removal(temp_pardir)


def local_file_fingerprint(path: os.PathLike, fingerprint_src: str) -> Tuple[str, Dict]:
    """Create unique fingerprint and metadata for local file.

    Fingerprint is the hash of the combination of:

        * resolved absolute ``path``
        * time of most recent content modification of ``path``.

    You can also choose to calculate the hash from the content of the file rather than its metadata.

    We also return a metadata dict that contains some information about the fingerprint.
    E.g., the path of the source file and when the hash was created.

    You can use this info to create a unique cashe directory for all the artifacts that are generated
    from ``path``.

    Args:
        path (os.PathLike): filepath to create metadata for.
        fingerprint_src (Optional[str]): how to calculate a unique hash for input_data. Options are

            * ``path``: hash the combination of resolved absolute path and the last modified timestamp.
            * ``content``: calculate a non-cryptographic hash from binary content of the file.

    Returns:
        unique fingerprint and metadata about `path`
    """
    _realpath = file_utils.realpath(path)
    if fingerprint_src == "path":
        hasher = Hasher()
        hasher.update(_realpath)
        hasher.update(Path(_realpath).stat().st_mtime)
        file_hash = hasher.hexdigest()
    elif fingerprint_src == "content":
        file_hash = file_utils.hash_file_bytes(path=path)
    else:
        msg = f"Supported values for 'fingerprint_src' are: 'path' and 'content'. Got: '{fingerprint_src}'"
        raise ValueError(msg)

    metadata_dict = {
        "source_path": Path(path).as_posix(),
        "source_realpath": _realpath,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if Path(path).exists():
        metadata_dict["last_modification_time"] = (
            Path(file_utils.realpath(path)).stat().st_mtime
        )
    return file_hash, metadata_dict


def hf_dataset_fingerprint(dataset: datasets.Dataset) -> Tuple[str, Dict]:
    """Creae unique fingerprint and metadata for ``datasets.Dataset`` object.

    Fingerprint is the hash of the combination of

        * original finger print of the dataset. This should be enough on its own but we do some extra work
        * parent directory that contains the arrow cache files that back the dataset.
          We include this because it has some information about split, source file, etc.

    We also return a metadata dict that contains info about the dataset that was used to compute the hash.

    Similar to :func:`local_file_fingerprint`, You can use this info to create a unique
    cash directory for all the artifacts that are generated from ``dataset``.

    Args:
        dataset (datasets.Dataset): dataset to create metadata for.

    Returns:
        unique fingerprint and metadata about dataset.
    """
    hasher = Hasher()
    hasher.update(dataset._fingerprint)
    if len(dataset.cache_files):
        _ds_cache_root = Path(dataset.cache_files[0]["filename"]).parent.as_posix()
        hasher.update(file_utils.realpath(_ds_cache_root))
    ds_hash = hasher.hexdigest()
    metadata_dict = asdict(dataset.info)
    return ds_hash, metadata_dict


def get_cache_pardir(
    artifact_content: str, artifact_type: str, cache_uuid: Optional[str] = None
) -> os.PathLike:
    """Create the parent directory holding cache files with specific content and type.

    The parent directory follows this format: ``CACHE_ROOT/ARTIFACT_TYPE/ARTIFACT_CONTENT``.
    ``ARTIFACT_TYPE`` and ``ARTIFACT_CONTENT`` are often the same as input arguments to this function with some exceptions.
    Exceptions are explained in comments in the code.

    Args:
        artifact_content (str): content of the artifacts (e.g., ``embedding``, ``grouped_qrel``, ``key_to_row_idx``, etc.).
            Basically asking what you want to cache.
        artifact_type (str): The type of artifacts. There are three possibilities

            * ``final``: these are the final results of the target operation
            * ``intermediate``: These are intermediate results and are just needed for a short time until the results of the next step are ready.
            * ``temp``: these are temporary cache files. Once this session ends, you won't be able to access them again.
              Also this is not synchronized across processes. You get a different cache pardir in each process.
            * ``user``: these are cache files saved and loaded by trove users and should not impact the internal functionality of the library.
        cache_uuid (Optional[str]): If provided, use ``cache_uuid`` whenever a random unique ID is needed.
            When your ``artifact_type='temp'`` or ``artifact_content='embedding'``, you can provide the same value for this argument
            from all processes to ensure they all have the same cache_pardir (e.g., across different processes)

    Returns:
        A cache directory that holds artifacts of specific type and content.
    """
    supported_art_types = ["final", "intermediate", "temp", "user"]

    parts = []
    if artifact_type == "user":
        parts.append("user")
    elif artifact_type == "final":
        parts.append("final")
    elif artifact_type == "intermediate":
        parts.append("intermediate_artifacts")
    elif artifact_type == "temp":
        parts.append("tmp")
        # For temporary results, create a random directory that changes everytime.
        # It is like deleting the directory. But, subsequent runs will not crash if we fail to clear the temporary files properly.
        if cache_uuid is not None:
            parts.append(cache_uuid)
        else:
            parts.append(config.PROCESS_UUID)
            # make sure the temporary cache pardir is removed before exit
            _register_temp_pardir_for_removal()
    elif artifact_type not in supported_art_types:
        msg = (
            f"`artifact_type` is not recognized."
            f" Supported artifact types are: {supported_art_types}. Got '{artifact_type}'"
        )
        raise ValueError(msg)

    parts.append(artifact_content)
    if artifact_content == "embedding":
        if cache_uuid is not None:
            parts.append(cache_uuid)
        else:
            parts.append(uuid4().hex)

    cache_pardir = hf_hub.cached_assets_path(
        library_name="trove",
        namespace=parts[0],
        subfolder=parts[1],
        assets_dir=config.TROVE_CACHE_DIR,
    )
    cache_pardir = cache_pardir.joinpath(*parts[2:])
    return cache_pardir.as_posix()


def get_cache_dir(
    input_data: Optional[Union[os.PathLike, datasets.Dataset]] = None,
    cache_pardir: Optional[os.PathLike] = None,
    artifact_content: Optional[str] = None,
    artifact_type: Optional[str] = None,
    cache_uuid: Optional[str] = None,
    write_metadata: bool = True,
    fingerprint_src: Optional[str] = "content",
    fingerprint: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> Path:
    """Creates a unique directory to save cache files generated for a specific input data.

    Cache files are organized in two levels. First is ``cache_pardir``, which organizes the cache files
    based on their type and content. See :func:`get_cache_pardir` for more info.
    Then, inside ``cache_pardir``, there is a unique directory for each source of input data.
    All cache files generated from a specific source of ``input_data`` should be saved in the corresponding cache directory.

    Args:
        input_data (Optional[Union[os.PathLike, datasets.Dataset]]): The data that was used to generate the cache.
            It can be either a local file or a ``datasets.Dataset`` instance. It is used to generate a unique fingerprint and some
            metadata about input data. You can provide the fingerprint and metadata dict directly instead of ``input_data``.
        cache_pardir (Optional[os.PathLike]): cache_pardir based on ``artifact_content`` and ``artifact_type``.
            If not provided, create it from ``artifact_content`` and ``artifact_type``.
        artifact_content (Optional[str]): see :func:`get_cache_pardir`
        artifact_type (Optional[str]): see :func:`get_cache_pardir`
        cache_uuid (Optional[str]): see :func:`get_cache_pardir`
        write_metadata (bool): If True, write a json file in ``cache_dir`` with some info about ``input_data``.
        fingerprint_src (Optional[str]): how to calculate a unique hash for input_data if it is a local path on disk.
            See :func:`local_file_fingerprint` for details.
        fingerprint (Optional[str]): a unique fingerprint for the data source that was used to generate the cache.
            if not provided, it is calculated from the value of ``input_data``.
        metadata (Optional[Dict]): a dictionary with some information about that helps identify the ``input_data`` that
            lead to this cache files (e.g. filepath, modification time, etc.). If not provided, it is calculated from ``input_data``.

    Returns:
        A unique cache directory to save artifacts generated by processing ``input_data``
    """
    if (input_data is None) == (fingerprint is None or metadata is None):
        msg = (
            "You should provide exactly one of the 'input_data' or '(fingerprint, metadata)' arguments."
            f" Got: 'input_data': '{input_data}', 'fingerprint': '{fingerprint}', 'metadata': '{metadata}'"
        )
        raise ValueError(msg)

    if input_data is None:
        if fingerprint is None or metadata is None:
            msg = (
                "You should provide both 'fingerprint' and 'metadata'."
                f" Got: 'fingerprint': '{fingerprint}', 'metadata': '{metadata}'"
            )
            raise ValueError(msg)

        _fingerprint = fingerprint
        _info = metadata
    else:
        if isinstance(input_data, datasets.Dataset):
            _fingerprint, _info = hf_dataset_fingerprint(input_data)
        else:
            _fingerprint, _info = local_file_fingerprint(
                path=input_data, fingerprint_src=fingerprint_src
            )

    if cache_pardir is None:
        if artifact_content is None or artifact_type is None:
            msg = "You should provide either `cache_pardir` or `artifact_content` and `artifact_type`."
            raise ValueError(msg)
        cache_pardir = get_cache_pardir(
            artifact_content=artifact_content,
            artifact_type=artifact_type,
            cache_uuid=cache_uuid,
        )

    cache_dir = Path(cache_pardir, _fingerprint)
    metadata_path = cache_dir / "source_metadata.json"

    try:
        cache_dir.mkdir(exist_ok=True, parents=True)
    except (FileExistsError, NotADirectoryError):
        raise ValueError(
            f"Corrupted cache folder: cannot create directory because of an existing file ({cache_dir})."
        )

    if write_metadata:
        with file_utils.easyfilelock(metadata_path):
            if not metadata_path.exists():
                with open(metadata_path, "w") as f:
                    json.dump(_info, f, indent=2)

    return cache_dir
