"""Some data transformations with polars that take a file as input and write to an output file.

These are just concerned with the main computation. Caching and other tasks should be done by the
user of these functions.
"""

import functools
import json
import os
import pickle as pkl
import sys
from pathlib import Path
from subprocess import run
from typing import Any, Dict


def call_script_fn(fn_name: str, kwargs: Dict[str, str]) -> None:
    """Call the file that the function is defined in as the main module with specific arguments.

    if you call ``call_script_fn('myfn', {'foo': 'bar'})``, this function calls the following shell function: ``$ python CURRENT_FILE myfn --foo bar``

    Together with the code in ``__main__`` of this file, this function allows us to call functions from this module
    in a separate process without using multiprocessing (e.g., fork). We want to avoid forking the
    process because polars has some problems with that in specific python versions.

    In general, we want to run polars functions in a separate process so the OS frees all the memory once the
    process is over. In early versions of polars, the memory was not released even after polar operations were done.
    This trick helps with that.

    Args:
        fn_name: the name of the function to call.
        kwargs: the keyword arguments to the target function.
    """
    cmd = list()
    cmd.append(Path(sys.executable).absolute().as_posix())
    cmd.append(Path(__file__).absolute().as_posix())
    cmd.append(fn_name)
    for k, v in kwargs.items():
        cmd.append(f"--{k}")
        cmd.append(f"{v}")
    run(cmd, check=True, cwd=Path.cwd().absolute().as_posix())


def csv_loader_kwargs(filepath: os.PathLike) -> Dict[str, Any]:
    """Automatically detect appropriate kwargs for ``pd.read_csv`` and similar tools.

    Args:
        filepath (os.PathLike): Path to delimiter separate file.

    Returns:
        A kwargs dict to be used like ``pd.read_csv(**kwargs)``
    """
    kwargs = dict(float_precision="high")

    # Inspect five lines to be more confident
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
        raise RuntimeError
    kwargs["sep"] = sep

    cols = [c.strip() for c in lines[0].strip().split(sep)]

    if len(cols) != 3:
        msg = "We only support delimiter separated files with three columns in this order (with or without headers): 'qid', 'docid', 'score'"
        raise NotImplementedError(msg)

    # common header names in IR csv/tsv files
    header_name_mapping = {
        "qid": "qid",
        "query_id": "qid",
        "query-id": "qid",
        "docid": "docid",
        "passage_id": "docid",
        "passage-id": "docid",
        "corpus-id": "docid",
        "score": "score",
    }

    matching_header_names = list(
        set(list(header_name_mapping.keys())).intersection(set(cols))
    )
    if len(matching_header_names) == 0:
        kwargs["header"] = None
    elif len(matching_header_names) == 3:
        kwargs["header"] = 0
    else:
        raise RuntimeError

    if kwargs["header"] is None:
        # This is the only supported format at the moment.
        kwargs["names"] = ["qid", "docid", "score"]
    else:
        kwargs["names"] = [header_name_mapping[i] for i in cols]

    return kwargs


def _key_to_row_indices_mapping_op(
    src_file: os.PathLike, dst_file: os.PathLike, id_field: str
) -> None:
    """Create key to row indices mapping using polars.

    Given a source file and a column name (``id_field``), this function creates a mapping from the values of the given column
    to the index of rows that contain that value (In fact, it is a list of row indices).

    ``src_file`` should be a JSONL file or an Arrow IPC file that contains a field named ``id_field``.
    The result written to ``dst_file`` is a pickle containing a dict from values of key field
    to list of indices of rows that contain those values.

    Args:
        src_file (os.PathLike): JSONL or arrow IPC file with key for each row
        dst_file (os.PathLike): a pickle file containing a dict from values of key field
            to list of indices of rows that contain those values.
        id_field (str): The name of the column that should be used as key
    """
    # logger.debug("Running Polars query to create key to row indices mapping.")

    import polars as pl

    if Path(src_file).suffix not in [".jsonl", ".arrow"]:
        msg = f"Expected a json lines or arrow IPC file (with '.jsonl' or '.arrow' suffix) as input. Got '{src_file}'"
        raise ValueError(msg)

    dst_file = Path(dst_file)
    if dst_file.suffix not in [".pkl", ".pickle"]:
        msg = f"We only support writing to pickle file. But, got: {dst_file.as_posix()}"
        raise ValueError(msg)

    if Path(src_file).suffix == ".jsonl":
        # check if the file is empty or has records
        empty_file = True
        with open(src_file, "r") as f:
            for line in f:
                if line.strip() != "":
                    empty_file = False
                    break

        if empty_file:
            # avoid errors if the file is empty
            rows_by_key = dict()
        else:
            df = pl.scan_ndjson(src_file)
            df = df.select(pl.col(id_field)).with_row_index()
            rows_by_key = df.collect().rows_by_key(id_field)
    elif Path(src_file).suffix == ".arrow":
        df = pl.scan_ipc(src_file, cache=False)
        df = df.select(pl.col(id_field)).with_row_index()
        rows_by_key = df.collect().rows_by_key(id_field)
    else:
        raise ValueError

    # in earlier versions of polars, if df had only one column, the output of polars was
    # a list of indices. In later versions, the output is a tuple of length one in such cases.
    # If it is a tuple, we need to flatten it before writing it to disk
    _sample_idx = next(iter(rows_by_key.values()))[0]
    if isinstance(_sample_idx, tuple):
        idx_is_tuple = True
    else:
        idx_is_tuple = False

    sorted_keys = sorted(list(rows_by_key.keys()), key=lambda x: str(x))
    sorted_dict = dict()
    for key in sorted_keys:
        _vals = rows_by_key[key]
        if idx_is_tuple:
            _vals = [i[0] for i in _vals]
        sorted_dict[str(key)] = sorted(_vals)

    dst_file.parent.mkdir(exist_ok=True, parents=True)
    with open(dst_file, "wb") as f:
        pkl.dump(sorted_dict, f)


@functools.wraps(_key_to_row_indices_mapping_op)
def key_to_row_indices_mapping_op(
    src_file: os.PathLike, dst_file: os.PathLike, id_field: str
) -> None:
    src_file = Path(src_file).absolute().as_posix()
    dst_file = Path(dst_file).absolute().as_posix()
    fn_kwargs = {"src_file": src_file, "dst_file": dst_file, "id_field": id_field}
    call_script_fn(fn_name="_key_to_row_indices_mapping_op", kwargs=fn_kwargs)


def _group_qrel_triplets_from_csv_op(
    src_file: os.PathLike, dst_file: os.PathLike
) -> None:
    """Group query triplets from csv file using polars.

    ``src_file`` should contain rows of (qid, docid, score) triplets.
    The only restriction on the structure of the file (e.g., num columns, headers, etc.)
    is that it should be supported by ``csv_loader_kwargs()`` function and once
    loaded using its returned arguments, it must have three columns named ``qid``, ``docid``, and ``score``.

    The results are written to a json lines file where each row is a dict of following format::

        {
            'qid': ...,
            'docid': [docid1, docid2, ...], # ID of related documents
            'score': [score1, score2, ...]  # similarity scores of related documents in the same order.
        }

    Args:
        src_file (os.PathLike): a csv/tsv file to read qrel triplets from.
        dst_file (os.PathLike): a target jsonl file to write the grouped results to.
    """
    import polars as pl

    dst_file = Path(dst_file)
    if dst_file.suffix != ".jsonl":
        msg = f"We only support writing to jsonl file. But, got: {dst_file.as_posix()}"
        raise ValueError(msg)

    # polars needs the directory to exist
    dst_file.parent.mkdir(exist_ok=True, parents=True)

    _loader_kwargs = csv_loader_kwargs(src_file)
    pl_loader_kwargs = {
        "has_header": _loader_kwargs.get("header", None) is not None,
        "separator": _loader_kwargs["sep"],
        "new_columns": _loader_kwargs["names"],
    }
    # Scan instead of read so if possible all the data is not read into memory
    df = pl.scan_csv(src_file, **pl_loader_kwargs, cache=False)
    # group by 'qid' column and collect docids and scores for each group in a list
    df = df.group_by("qid").agg(pl.col("docid"), pl.col("score"))
    # Run the actual query
    df = df.collect()
    df.write_ndjson(dst_file.as_posix())


@functools.wraps(_group_qrel_triplets_from_csv_op)
def group_qrel_triplets_from_csv_op(
    src_file: os.PathLike, dst_file: os.PathLike
) -> None:
    src_file = Path(src_file).absolute().as_posix()
    dst_file = Path(dst_file).absolute().as_posix()
    fn_kwargs = {"src_file": src_file, "dst_file": dst_file}
    call_script_fn(fn_name="_group_qrel_triplets_from_csv_op", kwargs=fn_kwargs)


def _group_qrel_triplets_from_sydir_corpus_op(
    src_file: os.PathLike, dst_file: os.PathLike
) -> None:
    """Group inferred qrel triplets from sydir docids.

    sydir docids are formatted as ``f"{qid}_l_{level}_d_{doc_idx}"``.
    We parse this and use ``level`` as the ``score`` field in the qrel triplets.

    see ``group_qrel_triplets_from_csv()`` function for the format of the output file.

    Args:
        src_file (os.PathLike): corpus file containing sydir synthetic documents in jsonl format.
            Each record is a document with either ``docid`` or ``_id`` field that follows sydir docid formatting.
        dst_file (os.PathLike): a target jsonl file to write the grouped results to.
    """
    import polars as pl

    dst_file = Path(dst_file)
    if dst_file.suffix != ".jsonl":
        msg = f"We only support writing to jsonl file. But, got: {dst_file.as_posix()}"
        raise ValueError(msg)

    # Find out the field name that holds docid values
    with open(src_file, "r") as f:
        _rec = json.loads(f.readline().strip())
    if "_id" in _rec and "docid" in _rec:
        msg = "The records contain both 'docid' and '_id' field. I cannot choose which one to use."
        raise RuntimeError(msg)
    elif "_id" in _rec:
        id_field = "_id"
    elif "docid" in _rec:
        id_field = "docid"
    else:
        msg = f"I could not find either '_id' or 'docid' fields in the given file: {src_file}"
        raise RuntimeError(msg)

    # polars needs the directory to exist
    dst_file.parent.mkdir(exist_ok=True, parents=True)

    # Scan instead of read so if possible all the data is not read into memory
    df = pl.scan_ndjson(src_file)
    # We only 'docid' field to create qrel triplets
    df = df.select(pl.col(id_field).alias("docid"))

    # we want to stick to polars builtin operations and avoid python functions since they are much faster
    # But the current version of polars cannot split text from right (does not have something like str.rsplit)
    # So, we reverse the text, split it, take a specific split, and reverse it again
    df = df.with_columns(
        pl.col("docid")
        .str.reverse()
        .str.splitn("_", n=5)
        .struct.field("field_4")  # this is the qid split
        .str.reverse()
        .alias("qid"),
        pl.col("docid")
        .str.reverse()
        .str.splitn("_", n=4)
        .struct.field("field_2")  # this is the score (i.e., level) split
        .str.reverse()
        .cast(pl.Int32)  # Makes computation more efficient
        .alias("score"),
    )
    # group by 'qid' column and collect docids and scores for each group in a list
    df = df.group_by("qid").agg(pl.col("docid"), pl.col("score"))
    # Run the actual query
    df = df.collect()
    df.write_ndjson(dst_file.as_posix())


@functools.wraps(_group_qrel_triplets_from_sydir_corpus_op)
def group_qrel_triplets_from_sydir_corpus_op(
    src_file: os.PathLike, dst_file: os.PathLike
) -> None:
    src_file = Path(src_file).absolute().as_posix()
    dst_file = Path(dst_file).absolute().as_posix()
    fn_kwargs = {"src_file": src_file, "dst_file": dst_file}
    call_script_fn(
        fn_name="_group_qrel_triplets_from_sydir_corpus_op", kwargs=fn_kwargs
    )


if __name__ == "__main__":
    fn_name = sys.argv[1]
    fn_kwargs = sys.argv[2:]
    if any(["=" in a for a in fn_kwargs]):
        msg = f"Only '--key' 'value' format is supported. But it is possible that args of '--key=value' formats were given. args: {sys.argv}"
        raise ValueError(msg)

    fn_kwargs = [akv for a in fn_kwargs for akv in a.split("=", maxsplit=1)]
    fn_kwargs = {
        key.lstrip("-"): value for key, value in zip(fn_kwargs[::2], fn_kwargs[1::2])
    }
    if fn_name == "_key_to_row_indices_mapping_op":
        _key_to_row_indices_mapping_op(**fn_kwargs)
    elif fn_name == "_group_qrel_triplets_from_csv_op":
        _group_qrel_triplets_from_csv_op(**fn_kwargs)
    elif fn_name == "_group_qrel_triplets_from_sydir_corpus_op":
        _group_qrel_triplets_from_sydir_corpus_op(**fn_kwargs)
    else:
        msg = (
            f"Function '{fn_name}' is not recognized. Only supported functions are"
            " ['_key_to_row_indices_mapping_op', '_group_qrel_triplets_from_csv_op', '_group_qrel_triplets_from_sydir_corpus_op']."
        )
        raise ValueError(msg)
