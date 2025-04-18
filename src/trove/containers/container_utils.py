import itertools
import os
import pickle as pkl
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import datasets
import pyarrow as pa
from datasets.formatting import query_table
from tqdm import tqdm

from .. import cache_manager, file_utils
from ..data import polars_transforms
from ..logging_utils import get_logger_with_config

logger, logging_conf = get_logger_with_config("trove")


def dataset_column_to_row_indices_mapping_op(
    dataset: datasets.Dataset,
    id_field: str,
    dst_file: os.PathLike,
    num_proc: Optional[int] = None,
) -> None:
    """Creates a mapping from values of a column (i.e., keys) to row indices for the given dataset.

    You should not use this function directly. Instead, use ``dataset_column_to_row_indices_mapping``.

    Writes the results to a pickle file that contains a dict
    from from values of ``id_field`` to a list of indices of rows that contain that value.

    We save them as a pickle file to avoid the cost of creating a ``dict`` object when loading it.

    Args:
        dataset (datasets.Dataset): dataset to create the index mapping for.
        id_field (str): name of the field to use as key.
        dst_file (os.PathLike): path to file where the resulting pickle file will be saved.
        num_proc (Optional[int]): number of parallel processes to use. Passed to ``datasets.Dataset.*`` methods.
    """
    dst_file = Path(dst_file)
    if dst_file.suffix not in [".pkl", ".pickle"]:
        msg = f"We only support writing to pickle file. But, got: {dst_file.as_posix()}"
        raise ValueError(msg)

    with file_utils.easyfilelock(dst_file):
        if not Path(dst_file).exists():
            keys_cache = cache_manager.get_cache_dir(
                input_data=dataset,
                artifact_content="key_to_row_idx_polars",
                artifact_type="intermediate",
                fingerprint_src="path",
            )
            # we do not need to lock this file for writing because it is only written to
            # in this function and while the lock for dst_cache is acquired.
            keys_cache = Path(keys_cache, "dataset_rows_with_key_only.arrow").as_posix()
            Path(keys_cache).parent.mkdir(exist_ok=True, parents=True)
            # we only need the keys. Ignore other columns
            key_ds = dataset.select_columns(id_field)

            batch_size = 1_000
            with pa.OSFile(keys_cache, "wb") as sink:
                with pa.ipc.new_file(sink, key_ds.features.arrow_schema) as writer:
                    # write in batches. It is more memory efficient
                    # adapted from here: https://github.com/huggingface/datasets/blob/01f91bae037c98f2e05456287bab21470adb8f07/src/datasets/arrow_dataset.py#L4951
                    for offset in tqdm(
                        range(0, len(key_ds), batch_size),
                        desc="Write dataset keys in arrow table",
                        disable=not logging_conf.is_debug(),
                    ):
                        sub_tab = query_table(
                            table=key_ds._data,
                            key=slice(offset, offset + batch_size),
                            indices=(
                                key_ds._indices if key_ds._indices is not None else None
                            ),
                        )
                        writer.write_table(sub_tab, 1_000)
            # It is recommended to use atomic_write right before starting the write operation
            # In this special case, it is okay to call it earlier because we know 'polars_transforms.key_to_row_indices_mapping_op'
            # does not do any processing/locking based on filepath and just writes to it.
            with file_utils.atomic_write(file=dst_file, root="parent") as tfile:
                polars_transforms.key_to_row_indices_mapping_op(
                    src_file=keys_cache, dst_file=tfile, id_field=id_field
                )


def dataset_column_to_row_indices_mapping(
    dataset: datasets.Dataset,
    id_field: str,
    num_proc: Optional[int] = None,
    load: bool = True,
) -> Union[Dict[Any, List[int]], os.PathLike]:
    """Creates a mapping from values of a column (i.e., keys) to row indices for the given dataset.

    It internally calls ``dataset_column_to_row_indices_mapping_op`` and does the same operation.
    However, this function saves the results to a unique cache file and reads from the cache file if available.
    It optionally returns the loaded mapping as a dictionary.

    Args:
        dataset (datasets.Dataset): dataset to create index mapping for.
        id_field (str): name of the field/column to use as key.
        num_proc (Optional[int]): number of parallel processes used. Arg passed to ``datasets.Dataset.*`` methods.
        load (bool): Whether to return the mapping object or return the path to the pickle file holding it.

    Returns:
        A mapping from the values of ``id_field`` column to indices of rows that contain that specific value.
        If ``load`` is ``False``, it returns the path to a pickle file containing the mapping instead of loading it.
    """
    logger.debug(
        f"Create key to row indices mapping. num records: {tqdm.format_sizeof(len(dataset))}"
    )

    cache_file = cache_manager.get_cache_dir(
        input_data=dataset,
        artifact_type="final",
        artifact_content="key_to_row_indices_mapping",
    )
    cache_file = Path(cache_file, "key2row_idx.pkl").as_posix()
    dataset_column_to_row_indices_mapping_op(
        dataset=dataset, id_field=id_field, dst_file=cache_file, num_proc=num_proc
    )
    if load:
        with open(cache_file, "rb") as f:
            key_to_indices = pkl.load(f)
    else:
        key_to_indices = cache_file

    return key_to_indices


def flatten_grouped_triplets(
    grouped_triplets: Dict[str, Union[List, Any]]
) -> List[Dict]:
    """Converts a dict of column values to a list of row records.

    You can use this to flatten the grouped qrels for each query.

    In ``grouped_triplets`` dictionary, all values of type ``list`` should be of the same length.
    Values that are not of type `list` are repeated for all rows.

    Example Input:

    .. code-block:: text

        {'a': [1, 2], 'b': ['x', 'y'], 'c': 'foo'}

    Example Output:

    .. code-block:: text

        [
            {'a': 1, 'b': 'x', 'c': 'foo'},
            {'a': 2, 'b': 'y', 'c': 'foo'}
        ]

    Args:
        grouped_triplets (Dict[str, Optional[Union[str, float, List]]]): dict from
            column names to column values (either a list of values or a constant values).

    Returns:
        a list row records.
    """
    qrel_recs = list()
    for grouped_recs in grouped_triplets:
        # Repeat values that are the same for all triplets to get a dict of lists
        _grecs = {
            k: v if isinstance(v, list) else itertools.repeat(v)
            for k, v in grouped_recs.items()
        }
        # Turn the dict of lists to list of dicts
        qrel_recs.extend(
            [
                dict(zip(ks, vs))
                for ks, vs in zip(
                    itertools.repeat(tuple(_grecs.keys())), zip(*_grecs.values())
                )
            ]
        )
    return qrel_recs


def filter_fn_wrapper(fn: Callable[[Dict[str, Union[str, float]]], bool]) -> Callable:
    """Wrapper that applies a given function that filters individual triplets over grouped
    triplets.

    You can use this function to apply ``fn`` to grouped triplets using ``datasets.Dataset.map``.

    Args:
        fn (Callable[[Dict[str, Union[str, float]]], bool]): callable that takes a dict
            of form ``{'qid': ..., 'docid': ..., 'score': ...}`` and returns a boolean.
            If returned true, we keep the triplet. And skip it otherwise.

    Returns:
        A function that applies ``fn`` over grouped triplets.
    """

    def filter_groups(groups: Dict[str, List]) -> Dict[str, List]:
        """Filtering a batch of grouped triplets."""
        groups_subset = {"qid": [], "docid": [], "score": []}
        for g_qid, g_docids, g_scores in zip(
            groups["qid"], groups["docid"], groups["score"]
        ):
            subset_docids = list()
            subset_scores = list()
            for _d, _s in zip(g_docids, g_scores):
                if fn(dict(qid=g_qid, docid=_d, score=_s)):
                    subset_docids.append(_d)
                    subset_scores.append(_s)
            # Only keep this group if it contains at least one triplet
            if len(subset_docids):
                groups_subset["qid"].append(g_qid)
                groups_subset["docid"].append(subset_docids)
                groups_subset["score"].append(subset_scores)
        return groups_subset

    return filter_groups
