"""Utilities for transforming data structure and format with support for caching."""

import os
from pathlib import Path
from typing import Callable, Optional

import datasets

from .. import cache_manager, file_utils
from ..logging_utils import get_logger_with_config
from . import polars_transforms

logger, logging_conf = get_logger_with_config("trove")


def _group_qrel_triplets(
    src_file: os.PathLike,
    dst_file: os.PathLike,
    groupby_op: Callable[[os.PathLike, os.PathLike], None],
    num_proc: Optional[int] = None,
) -> None:
    """Read qrel triplets from file and Group them by qid field.

    At the moment, we only support reading qrel triplets from CSV files or infer them from sydir docids.

    We use ``groupby_op`` to read qrels and group them.
    Additionally, we take the output of these functions and cast qids and docids to ``str``.

    Args:
        src_file (os.PathLike): File to read or infer qrels triplets from.
        dst_file (os.PathLike): file to write the resulting grouped qrels to.
        groupby_op (Callable[[os.PathLike, os.PathLike], None]): a callable that does
            the actual groupby operation. It should take two keyword arguments, ``src_file``
            and ``dst_file``. It should read the qrels from ``src_file`` and write the grouped
            qrels to ``dst_file``.
        num_proc (Optional[int]): arg to ``datasets.Dataset.*`` methods
    """
    dst_file = Path(dst_file)
    if dst_file.suffix != ".jsonl":
        msg = f"We only support writing to jsonl file. But, got: {dst_file.as_posix()}"
        raise ValueError(msg)

    with file_utils.easyfilelock(dst_file):
        if not Path(dst_file).exists():
            # if dst_file does not exist, we should first use groupby_op to create a jsonl
            # file containing the output and then cast data types of that output if necessary
            dst_cache = cache_manager.get_cache_dir(
                input_data=src_file,
                artifact_content="grouped_qrel_triplets_polar",
                artifact_type="intermediate",
                fingerprint_src="path",
            )
            # output from groupby_op which might contain wrong data types
            dst_cache = Path(dst_cache, "grouped_qrels_pl.jsonl").as_posix()
            # we do not need to lock this file for writing because it is only written to
            # in this function and while the lock for dst_cache is acquired.
            # We do NOT need an atomic_write context for groupby_op it is always overwritten if dst_file
            # does not exists. And if dst_file exists, we successfully created the final file
            # and do not care if ouptut of groupby_op is corrupted later or not
            groupby_op(src_file=src_file, dst_file=dst_cache)

            logger.debug(
                "Load grouped qrels by HF dataset to potentially cast dtypes and save output as json lines."
            )
            # Read polars results with hf datasets to cast qid and docid if necessary and write the final results
            ds = datasets.load_dataset("json", data_files=dst_cache, split="train")

            # Make sure the IDs are of type 'str'
            new_features = ds.features.copy()
            if new_features["qid"] != datasets.Value("string") or new_features[
                "docid"
            ].feature != datasets.Value("string"):
                logger.debug("Cast 'qid' and 'docids' to 'str'")

                new_features["qid"] = datasets.Value("string")
                new_features["docid"].feature = datasets.Value("string")

                ds = ds.cast(new_features, num_proc=num_proc)

            with file_utils.atomic_write(file=dst_file, root="parent") as tfile:
                # Write results as json lines file
                ds.to_json(tfile)


def group_qrel_triplets_from_csv(
    src_file: os.PathLike,
    dst_file: os.PathLike,
    num_proc: Optional[int] = None,
) -> None:
    """Read qrel triplets from csv file and write them in grouped qrel format.

    see :func:`~trove.data.polars_transforms.group_qrel_triplets_from_csv_op` for the expected format
    of ``src_file`` and the format of the output ``dst_file``.

    Args:
        src_file (os.PathLike): csv/tsv file to read qrels from.
        dst_file (os.PathLike): output file to write the resulting grouped qrels
        num_proc (Optional[int]): max number of parallel process to use for data processing
    """
    _group_qrel_triplets(
        src_file=src_file,
        dst_file=dst_file,
        groupby_op=polars_transforms.group_qrel_triplets_from_csv_op,
        num_proc=num_proc,
    )


def group_qrel_triplets_from_sydir_corpus(
    src_file: os.PathLike,
    dst_file: os.PathLike,
    num_proc: Optional[int] = None,
) -> None:
    """Create qrel triplets from sydir document IDs and write them in grouped qrel format.

    see :func:`~trove.data.polars_transforms.group_qrel_triplets_from_sydir_corpus_op` for the expected format
    of ``src_file`` and the format of the output ``dst_file``.

    Args:
        src_file (os.PathLike): sydir corpus file with document IDs to use to create qrel triplets
        dst_file (os.PathLike): output file to write the resulting grouped qrels
        num_proc (Optional[int]): max number of parallel process to use for data processing
    """
    _group_qrel_triplets(
        src_file=src_file,
        dst_file=dst_file,
        groupby_op=polars_transforms.group_qrel_triplets_from_sydir_corpus_op,
        num_proc=num_proc,
    )
