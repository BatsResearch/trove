"""Loader functions to read various file formats."""

import json
import os
import pickle as pkl
import re
from pathlib import Path
from typing import List, Optional

import datasets

from .. import cache_manager, file_utils
from ..data import file_transforms
from ..logging_utils import get_logger_with_config
from . import file_reader

logger, logging_conf = get_logger_with_config("trove")


@file_reader.register_loader("qrel")
def qrel_from_grouped_triplets(
    filepath: os.PathLike, num_proc: Optional[int] = None
) -> datasets.Dataset:
    """Load grouped qrel triplets from JSONL file.

    Each line should have three fields::

        {
            'qid': '...',
            'docid': ['docid1', 'docid2', ...],
            'score': [score1, score2, ...]
        }

    Args:
        filepath (os.PathLike): Path to a JSONL file with grouped qrel triplets
        num_proc (Optional[int]): Max number of processes when reading and pre-processing the data.

    Returns:
        huggingface ``datasets.Dataset`` with ``'qid'``, ``'docid'``, and ``'score'`` columns of ``str``, ``List[str]``, and ``List[float]`` dtype, respectively.
    """
    # check the given file contains grouped qrels in jsonl format
    if not Path(filepath).exists():
        return
    if Path(filepath).suffix != ".jsonl":
        return
    with open(filepath, "r") as f:
        rec = json.loads(f.readline())
    if not {"qid", "docid", "score"}.issubset(set(list(rec.keys()))):
        return
    if not (
        isinstance(rec["qid"], str)
        and isinstance(rec["docid"], list)
        and isinstance(rec["score"], list)
    ):
        return

    logger.debug("Read qrel from grouped triplets jsonl file")
    ds = datasets.load_dataset(
        "json", data_files=filepath, split="train", num_proc=num_proc
    )
    return ds


@file_reader.register_loader("qrel")
def qrel_from_csv(
    filepath: os.PathLike, num_proc: Optional[int] = None
) -> datasets.Dataset:
    """Load qrel file from a delimiter separated file.

    It only supports a file with exactly three columns: ``qid``, ``docid``, ``score``
    If the headers are missing, it is assumed the columns are in the following order: ``['qid', 'docid', 'score']``.

    Args:
        filepath (os.PathLike): path to CSV/TSV file
        num_proc (Optional[int]): Max number of processes when reading and pre-processing the data.

    Returns:
        huggingface ``datasets.Dataset`` with ``qid``, ``docid``, and ``score`` columns of ``str``, ``List[str]``, and ``List[float]`` dtype, respectively.
    """
    # check if it is a csv or tsv file
    if not Path(filepath).exists():
        return
    if Path(filepath).suffix not in [".csv", ".tsv"]:
        return
    logger.debug("Read qrel from csv")

    # Inspect first five lines to figure out the delimiter
    lines = list()
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            lines.append(line)
            if i > 10:
                break
    # Check for common delimiters
    if all("\t" in line.strip() for line in lines):
        sep = "\t"
    elif all("," in line.strip() for line in lines):
        sep = ","
    elif all(" " in line.strip() for line in lines):
        sep = " "
    else:
        msg = "Could not identify delimiter. It is not any of the supported ',' '\t' 'SPACE' characters."
        raise RuntimeError(msg)

    if any(len(line.strip().split(sep)) != 3 for line in lines):
        msg = (
            "Only csv/tsv files with exactly three columns ('qid', 'docid', 'score') are supported."
            f" number of columns in file: {len(lines[0].strip().split(sep))}"
        )
        raise RuntimeError(msg)

    cache_file = cache_manager.get_cache_dir(
        input_data=filepath, artifact_type="final", artifact_content="grouped_qrels"
    )
    cache_file = Path(cache_file, "qrel.jsonl").as_posix()

    file_transforms.group_qrel_triplets_from_csv(
        src_file=filepath, dst_file=cache_file, num_proc=num_proc
    )
    ds = datasets.load_dataset("json", data_files=cache_file, split="train")
    return ds


@file_reader.register_loader("qrel")
def qrel_from_pickle(
    filepath: os.PathLike, num_proc: Optional[int] = None
) -> datasets.Dataset:
    """Load qrel triplets from pickle files.

    The file is expected to contain a single object of type dict.
    ``object[qid][docid]`` is the corresponding score for query ``qid`` and document ``docid``.
    ``qid`` and ``docid`` should be of type `str`.


    Args:
        filepath (os.PathLike): pickle file to load.
        num_proc (Optional[int]): Max number of processes when reading and pre-processing the data.

    Returns:
        huggingface datasets.Dataset with ``qid``, ``docid``, and ``score`` columns of ``str``, ``List[str]``, and ``List[float]`` dtype, respectively.
    """
    if not Path(filepath).exists():
        return
    if Path(filepath).suffix not in [".pkl", ".pickle"]:
        return

    logger.debug("Read qrel from pickle.")

    # Cache results to both reduce memory usage by using datasets arrow tables
    # And also avoid duplicate computation next time
    cache_file = cache_manager.get_cache_dir(
        input_data=filepath, artifact_type="final", artifact_content="grouped_qrels"
    )
    cache_file = Path(cache_file, "qrel.jsonl").as_posix()

    with file_utils.easyfilelock(cache_file):
        if not Path(cache_file).exists():
            with open(filepath, "rb") as f:
                dump = pkl.load(f)

            # Check the data structure and dtypes
            if (
                not isinstance(dump, dict)
                or not isinstance(next(iter(dump.keys())), str)
                or not isinstance(next(iter(dump.values())), dict)
                or not isinstance(next(iter(next(iter(dump.values())).keys())), str)
                or not isinstance(
                    next(iter(next(iter(dump.values())).values())), (int, float)
                )
            ):
                msg = (
                    "Data in pickle file is not correctly formatted. It is expected to be a nested dictionary, where dump[qid][docid]=score(qid, docid)."
                    "both qid and docid are expected to be of type 'str' and their similarity should be of type 'int' or 'float'"
                )
                raise RuntimeError(msg)

            with file_utils.atomic_write(file=cache_file, root="parent") as tfile:
                with open(tfile, "w") as f:
                    for i, (qid, qdata) in enumerate(dump.items()):
                        docids, scores = list(zip(*qdata.items()))
                        row = dict(qid=qid, docid=list(docids), score=list(scores))
                        f.write(json.dumps(row) + "\n")
                        if not i % 1000:
                            f.flush()

    ds = datasets.load_dataset(
        "json", data_files=cache_file, split="train", num_proc=num_proc
    )
    return ds


@file_reader.register_loader("qrel")
def qrel_from_tevatron_training_data(
    filepath: os.PathLike, num_proc: Optional[int] = None
) -> datasets.Dataset:
    """Convert a tevatron training file to qrel triplets.

    The data is expected to be a JSONL file with the structure that `tevatron <https://github.com/texttron/tevatron>`_ uses for training files.
    Each record (i.e., line) should be::

        {
            'query_id': 'target query id',
            'query': 'text of the query',
            'positive_passages': [{'docid': 'od of pos doc', 'title': 'title of pos doc', 'text': 'text of pos doc'}, ..., ...],
            'negative_passages': [{'docid': 'od of neg doc', 'title': 'title of neg doc', 'text': 'text of neg doc'}, ..., ...]
        }

    When creating (qid, docid, score) triplets, we give a score of ``0`` to all negative passages and a score of ``1`` to all positive passages.

    Args:
        filepath (os.PathLike): path to JSONL file in tevatron training format.
        num_proc (Optional[int]): arg to ``datasets.Dataset.map``

    Returns:
        huggingface datasets.Dataset with ``qid``, ``docid``, and ``score`` columns of ``str``, ``List[str]``, and ``List[float]`` dtype, respectively.
    """
    # check if it is tevatron training file
    if not Path(filepath).exists():
        return
    if Path(filepath).suffix != ".jsonl":
        return
    with open(filepath, "r") as f:
        rec = json.loads(f.readline())
    if set(list(rec.keys())) != {
        "query_id",
        "query",
        "positive_passages",
        "negative_passages",
    }:
        return
    if not (
        isinstance(rec["query_id"], str)
        and isinstance(rec["query"], str)
        and isinstance(rec["positive_passages"], list)
        and isinstance(rec["negative_passages"], list)
    ):
        return

    def process_row(row):
        new_row = dict()
        new_row["qid2"] = row["query_id"]
        new_row["docid2"] = list()
        new_row["score2"] = list()

        new_row["docid2"].extend([n["docid"] for n in row["negative_passages"]])
        new_row["score2"].extend([0] * len(row["negative_passages"]))

        new_row["docid2"].extend([p["docid"] for p in row["positive_passages"]])
        new_row["score2"].extend([1] * len(row["positive_passages"]))
        return new_row

    logger.debug("Read tevatron training file and convert to grouped qrel format.")

    ds = datasets.load_dataset("json", data_files=filepath, split="train")
    ds = ds.map(process_row, remove_columns=ds.column_names, num_proc=num_proc)
    ds = ds.rename_columns({"qid2": "qid", "docid2": "docid", "score2": "score"})
    return ds


@file_reader.register_loader("qrel")
def qrel_from_sydir_corpus(
    filepath: os.PathLike, num_proc: Optional[int] = None
) -> datasets.Dataset:
    """Infer qrel triplets from sydir docids.

    sydir docids are formatted as ``f"{qid}_l_{level}_d_{doc_idx}"``.
    We parse this and use ``level`` as the ``score`` field in the qrel triplets.

    Args:
        filepath (os.PathLike): sydir corpus file
        num_proc (Optional[int]): Max number of processes when reading and pre-processing the data.

    Returns:
        huggingface datasets.Dataset with ``qid``, ``docid``, and ``score`` columns of ``str``, ``List[str]``, and ``List[float]`` dtype, respectively.
    """
    # check if it is a sydir corpus file
    if not Path(filepath).exists():
        return
    if Path(filepath).suffix != ".jsonl":
        return
    with open(filepath, "r") as f:
        rec = json.loads(f.readline())
    if set(list(rec.keys())) != {"_id", "title", "text"}:
        return
    if "_id" in rec:
        rec_id = rec["_id"]
    elif "docid" in rec:
        rec_id = rec["docid"]
    if not isinstance(rec_id, str):
        msg = "All record IDs should be of type 'str'. Got: {type(rec_id)}"
        raise TypeError(msg)
    # Check if the ID follows sydir docid pattern.
    matches = re.findall(r".*?_l_([0-9].*)_d_([0-9].*)", rec_id)
    # There is only one such match in sydir docid format
    if len(matches) != 1:
        return
    try:
        # If it is a sydir docid, the extracted pieces should be both integers.
        _ = int(matches[0][0])
        _ = int(matches[0][1])
    except:
        return

    logger.debug("Read qrel from synthetic sydir corpus file.")
    cache_file = cache_manager.get_cache_dir(
        input_data=filepath, artifact_type="final", artifact_content="grouped_qrels"
    )
    cache_file = Path(cache_file, "qrel.jsonl").as_posix()

    file_transforms.group_qrel_triplets_from_sydir_corpus(
        src_file=filepath, dst_file=cache_file, num_proc=num_proc
    )
    ds = datasets.load_dataset("json", data_files=cache_file, split="train")
    return ds


@file_reader.register_loader("qid")
def qids_from_queries_jsonl(
    filepath: os.PathLike, num_proc: Optional[int] = None
) -> List[str]:
    """Load qids from the original `queries.jsonl` files.

    It expect an ``_id`` field in each record of the JSONL file.

    Args:
        filepath (os.PathLike): ``queries.jsonl`` file to read.
        num_proc (Optional[int]): Max number of processes when reading and pre-processing the data.

    Returns:
        A list of query IDs
    """
    # check that the given file contains original query records
    if not Path(filepath).exists():
        return
    if Path(filepath).suffix != ".jsonl":
        return
    with open(filepath, "r") as f:
        rec = json.loads(f.readline())
    if set(list(rec.keys())) != {"_id", "text"}:
        # it has other fields that are not usually found in queries.jsonl files
        # or it is missing the two fields that are usually found in queries.jsonl files.
        return

    logger.debug("Read list of qids from original queries.jsonl format.")

    ds = datasets.load_dataset("json", data_files=filepath, split="train")

    # Make sure the IDs are of type 'str'
    new_features = ds.features.copy()
    if new_features["_id"] != datasets.Value("string"):
        logger.debug("Cast '_id' field to 'str' dtype.")

        new_features["_id"] = datasets.Value("string")
        ds = ds.cast(new_features, num_proc=num_proc)
    return ds["_id"]
