import os
from pathlib import Path
from typing import Callable, List, Optional

import datasets
import numpy as np

from ..logging_utils import get_logger_with_config

logger, logging_conf = get_logger_with_config("trove")

# hold pointer to functions that read the data from file
# keys are the type of data that the function returns
READERS_REGISTRY = {"qrel": [], "qid": [], "record": []}


class FileLoaderNotFoundError(Exception):
    """No loader found for the given file."""


def available_loaders() -> None:
    """Reports what loader functions are available for each output type."""
    print("Available file loaders for each output type.")
    for output_type, loaders in READERS_REGISTRY.items():
        print(" " * 2, f"Output type: <{output_type}>")
        for fn in loaders:
            fn_fullname = fn.__module__ + "." + fn.__name__
            print(" " * 4, fn_fullname)


def register_loader(output: str) -> Callable:
    """A decorator to register functions that read the data from file.

    Later we loop through the registered functions to find one that can load the desired data from a given file.
    All file reader functions should take two keyword arguments: ``'filepath'`` and ``'num_proc'``.

        * ``'filepath'`` is the file that should be loaded.

        * ``'num_proc'`` is the number of processes that the loader can launch in parallel to load and preprocess the data.

    Each reader function should first check if it can load the given file.
    The loader functions should return ``None`` if it cannot read the given file (e.g., if reader can load CSV but the file is in JSON format).

    We go through all the readers until we find one that can load the file.
    So, **make sure the initial check for ability to read the file is fast.**
    In worst case scenario, it is possible that all readers need to do this check before we find the correct loader.

    The expected output for each loader is as following:

        - ``'qrel'`` : instance of huggingface ``datasets.Dataset`` with ``'qid'``, ``'docid'``, and ``'score'`` columns.

            * ``'qid'`` is of type `str` and represents the query id for the record

            * ``'docid'`` is a list of 'str' values (``List[str]``),
              where each item is the id of one related document for this query.

            * ``'score'`` is a list of `int` or `float` (``List[Union[int, float]]``),
              where ``datasets.Dataset[i]['score'][idx]`` is the similarity score betwen
              query ``'qid'`` and document ``datasets.Dataset[i]['docid'][idx]``.

        - ``'qid'`` : a list of query IDs of type `str` (``List[str]``).
          The returned query IDs **MUST** be unique without duplicate query IDs in the list.

        - ``'record'`` : instance of huggingface ``datasets.Dataset``.
          It just needs to load the records in the given file as-is without any further processing.
          It is recommended to keep the records in the same order as they appear in the given file.
          There are no restrictions on what the columns or their data type are.

    Args:
        output (str): The data that the loader returns. Accepted values are ``'qrel'``, ``'qid'``, and ``'record'``.

    Returns:
        A wrapper that registers the given loader function under ``'output'`` key.
    """
    global READERS_REGISTRY
    if output not in READERS_REGISTRY:
        msg = (
            f"'output' of the file reader should one of '{list(READERS_REGISTRY.keys())}'."
            f" Got '{output}'"
        )
        raise ValueError(msg)

    def _registrar(func: Callable) -> Callable:
        # prioritize user-defined functions over the ones defined by the library.
        if "trove" in func.__module__.split("."):
            READERS_REGISTRY[output].append(func)
        else:
            READERS_REGISTRY[output].insert(0, func)
        return func

    return _registrar


def load_qrel(
    filepath: os.PathLike, num_proc: Optional[int] = None
) -> datasets.Dataset:
    """Load qrel triplets from file.

    It loads grouped triplets of (qid, docid, score) from files.
    It loops through all registered qrel readers until it finds one that can load the given file.
    It returns huggingface ``datasets.Dataset`` with ``'qid'``, ``'docid'``, and ``'score'`` columns
    of ``str``, ``List[str]``, and ``List[float]`` dtype, respectively.
    For each record, ``'docid'`` is a list of document ids that are related to ``'qid'``
    and ``'score'`` is the list of similarity score between ``'qid'`` and each document in ``'docid'``.

    Args:
        filepath (os.PathLike): file to read qrels from
        num_proc (Optional[int]): Max number of processes when reading and pre-processing the data.

    Returns:
        huggingface datasets.Dataset with ``'qid'``, ``'docid'``, and ``'score'`` columns of ``str``, ``List[str]``, and ``List[float]`` dtype, respectively.
    """
    global READERS_REGISTRY
    output = None
    for reader_fn in READERS_REGISTRY["qrel"]:
        output = reader_fn(filepath=filepath, num_proc=num_proc)
        if output is not None:
            break

    if output is None:
        msg = f"Could not find a qrel loader function that can load this file:\n{Path(filepath).as_posix()}"
        raise FileLoaderNotFoundError(msg)

    # Since the readers can also be defined by the user, we ensure that the output conforms to
    # the expected structure and data types.

    if not isinstance(output, datasets.Dataset):
        msg = (
            f"The qrel reader function should return an instance of 'datasets.Dataset'."
            f" Got output of type: '{type(output)}'"
        )
        raise TypeError(msg)

    if set(output.column_names) != {"qid", "docid", "score"}:
        msg = (
            f"qrel reader's output should contain exactly three columns: 'qid', 'docid', and 'score'"
            f" Got: {output.column_names}"
        )
        raise ValueError(msg)

    # Check the data type of the columns
    feats = output.features
    if not (
        isinstance(feats["qid"], datasets.Value)
        and feats["qid"].dtype == "string"
        and isinstance(feats["docid"], datasets.Sequence)
        and isinstance(feats["docid"].feature, datasets.Value)
        and feats["docid"].feature.dtype == "string"
        and isinstance(feats["score"], datasets.Sequence)
        and isinstance(feats["score"].feature, datasets.Value)
        and np.issubdtype(feats["score"].feature.pa_type.to_pandas_dtype(), np.number)
    ):
        expected_dtype = {
            "qid": "str",
            "docid": "List[str]",
            "score": "List[Union[int, float]]",
        }
        msg = (
            f"qrel reader's output should have the following schema (or its equivalents in datasets or pyarrow): {expected_dtype}."
            f" Got: {feats}"
        )
        raise TypeError(msg)

    return output


def load_qids(filepath: os.PathLike, num_proc: Optional[int] = None) -> List[str]:
    """Load a list of unique qids from file.

    It loops through all registered qid readers until it finds one that can load the given file.
    If there is no dedicated qid reader that can load this file,
    it assumes it is a file with qrel triplets and returns the list of unique qids in qrel triplets.

    Args:
        filepath (os.PathLike): file to read qids from
        num_proc (Optional[int]): Max number of processes when reading and pre-processing the data.

    Returns:
        A list of unique query IDs (``List[str]``)
    """
    logger.debug("Reading a list of qids")

    global READERS_REGISTRY
    output = None
    for reader_fn in READERS_REGISTRY["qid"]:
        output = reader_fn(filepath=filepath, num_proc=num_proc)
        if output is not None:
            break

    if output is None:
        qrels = load_qrel(filepath=filepath, num_proc=num_proc)
        if qrels is not None:
            if qrels._indices is not None:
                # Dataset is subsampled and the underlying arrow table is not the same as data that dataset uses.
                output = list(set(qrels["qid"]))
            else:
                # Using the underlying pyarrow table is faster/more efficient in some cases
                output = qrels._data.table["qid"].unique().to_pylist()

    if output is None:
        msg = f"Could not find a qid or qrel loader function that can load this file:\n{Path(filepath).as_posix()}"
        raise FileLoaderNotFoundError(msg)

    if len(output) != len(set(output)):
        msg = "The loaded list of qids contains duplicates, which is not allowed."
        raise RuntimeError(msg)

    if not isinstance(output, list):
        msg = f"Expected a output of type 'List'. Got: '{type(output)}'"
        raise TypeError(msg)
    if not isinstance(output[0], str):
        msg = f"qids must be of type 'str'. Got: '{type(output[0])}'"
        raise TypeError(msg)

    return output


def load_records(
    filepath: os.PathLike, num_proc: Optional[int] = None
) -> datasets.Dataset:
    """Load records/rows from file.

    It loops through all registered record readers until it finds one that can load the given file.
    It returns an instance of huggingface ``datasets.Dataset`` that contain the records in the file
    without any further processing or modification.

    Load records/rows from a given file into an instance of ``datasets.Dataset``

    Args:
        filepath (os.PathLike): file to read the records from.
        num_proc (Optional[int]): Max number of processes when reading and pre-processing the data.

    Returns:
        an instance of ``datasets.Dataset`` containing records from the given file.
    """
    global READERS_REGISTRY
    output = None
    for reader_fn in READERS_REGISTRY["record"]:
        output = reader_fn(filepath=filepath, num_proc=num_proc)
        if output is not None:
            break

    if output is None:
        msg = f"Could not find a record loader function that can load this file:\n{Path(filepath).as_posix()}"
        raise FileLoaderNotFoundError(msg)

    if not isinstance(output, datasets.Dataset):
        msg = (
            f"The record reader function should return an instance of 'datasets.Dataset'."
            f" Got output of type: '{type(output)}'"
        )
        raise TypeError(msg)
    return output
