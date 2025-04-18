import copy
import itertools
import json
import os
import pickle as pkl
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import datasets
import numpy as np
from datasets.fingerprint import Hasher
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .. import cache_manager, containers, file_utils
from ..containers import MaterializedQRel, MaterializedQRelConfig
from ..file_utils import JSONLinesWriter
from ..logging_utils import get_logger_with_config, rpath
from .data_args import DataArguments
from .ir_encoding_dataset import EncodingDataset

logger, logging_conf = get_logger_with_config("trove")


EVAL_CACHE_FILE_NAMES = {
    "file_info_for_encoding": "file_info_for_encoding.json",
    "global_qrel": "qrels_with_global_id.pkl",
    "local_qrel": "qrels_with_local_id.pkl",
}


class MultiLevelDataset(Dataset):
    def __init__(
        self,
        data_args: DataArguments,
        format_query: Callable[[str, Optional[str]], str],
        format_passage: Callable[[str, str, Optional[str]], str],
        qrel_config: Optional[
            Union[MaterializedQRelConfig, List[MaterializedQRelConfig]]
        ] = None,
        eval_cache_path: Optional[os.PathLike] = None,
        train_cache_path: Optional[os.PathLike] = None,
        data_args_overrides: Optional[Dict[str, Any]] = None,
        num_proc: Optional[int] = None,
    ) -> None:
        """IR training dataset with multiple levels of relevancy (supports more than two levels).

        * Collection of related documents for queries are created from one or more ``qrel_config``
        * If there are multiple collections and a passage shows up in multiple collections,
          the data from the last collection takes precedence
          (i.e., whatever record the object corresponding to
          ``qrel_config[-1]`` returns will be used for that specific passage)

        Args:
            data_args (DataArguments): general arguments for loading and processing the data
            format_query (Callable[[str, Optional[str]], str]): A callable that takes the query
                text and optionally the dataset name and returns the formatted query text for the modle.
            format_passage (Callable[[str, str, Optional[str]], str]): A callable that takes
                the passage text and title and dataset name and returns the formatted passage text for the model.
            data_args_overrides (Optional[Dict[str, Any]]): A mapping from a subset of ``DataArguments`` attribute names to
                their new values. These key values override the corresponding attributes in ``data_args`` argument.
                It is useful if you want to create multiple datasets from the same ``DataArguments`` instance
                but make small changes for each dataset without creating new ``DataArguments`` instances.
            qrel_config (Optional[Union[MaterializedQRelConfig, List[MaterializedQRelConfig]]]):
                Config for one or more collections of queries, passages, and the relation between them.
                The combination of these collections will make up the content of this dataset.
            eval_cache_path (Optional[os.PathLike]): **DO NOT USE**. For internal operations only and not stable.
                If given, create a dataset only for evaluation from cache files in this directory.
                This is much more memory efficient compared to creating the dataset on-the-fly.
                You should use :meth:`export_and_load_eval_cache` method to take advantage of this.
            train_cache_path (Optional[os.PathLike]): **DO NOT USE**. For internal operations only and not stable.
                If given, create a dataset only for training from this cache file.
                This is much more memory efficient compared to creating the dataset on-the-fly.
                You should use :meth:`export_and_load_train_cache` method to take advantage of this.
            num_proc (Optional[int]): arg to to methods like ``datasets.Dataset.*``
        """
        super().__init__()
        # A unique fingerprint for the underlying data for this dataset.
        # Note that two datasets with the same fingerprint are only based on the same data
        # but can generate different samples. see docstring for self.fingerprint for details.
        # do not use this directly. Instead use 'self.fingerprint' attribute
        self._unique_fingerprint = None
        # some metadata about this dataset to be saved along with any cache files if necessary
        # Do not use this directly. Instead, use 'self.info' attribute
        self._info = None
        # If true, MultiLevelDataset[i] returns the training group (i.e., output of
        # self.create_group() which just returns the related docs) for each query rather than a training sample
        # this is useful for iterating over the underlying data for each dataset instance.
        self._iter_over_groups = False

        if (
            sum(i is not None for i in [qrel_config, eval_cache_path, train_cache_path])
            != 1
        ):
            msg = (
                "You should pass exactly on of the 'qrel_config', 'eval_cache_path', or 'train_cache_path' arguments."
                f" Got: 'qrel_config': '{qrel_config}', 'eval_cache_path': '{eval_cache_path}', 'train_cache_path': '{train_cache_path}'"
            )

            raise ValueError(msg)

        # make a deepcopy so we do not impact operations outside of this class.
        data_args = copy.deepcopy(data_args)
        if data_args_overrides is not None:
            for attr, val in data_args_overrides.items():
                if not hasattr(data_args, attr):
                    # We should not add new attributes
                    msg = f"'{attr}' is not a valid attribute name for 'DataArguments'"
                    raise ValueError(msg)
                setattr(data_args, attr, val)
        self.args = data_args

        if self.args.passage_selection_strategy not in [
            None,
            "random",
            "least_relevant",
            "most_relevant",
        ]:
            msg = (
                f"We only support [None, 'random', 'least_relevant', 'most_relevant'] for `passage_selection_strategy`."
                f" But, got: '{self.args.passage_selection_strategy}'"
            )
            raise ValueError(msg)

        self.format_query = format_query
        self.format_passage = format_passage

        # by default, the training groups should be created on-the-fly
        self.create_group = self.create_group_on_the_fly

        self.train_records = None
        self.qrel_with_global_ids_cache = None
        self.qrel_with_local_ids_cache = None
        self.file_info_for_encoding_cache = None
        if train_cache_path is not None:
            logger.info(
                f"Creating MultiLevelDataset from cached training groups in {rpath(train_cache_path)}."
            )
            if Path(train_cache_path).suffix != ".jsonl":
                msg = f"Only supported format for 'train_cache_path' is json lines. Got: {train_cache_path}"
                raise ValueError(msg)
            self.train_records = datasets.load_dataset(
                "json", data_files=Path(train_cache_path).as_posix(), split="train"
            )
            self.create_group = self.create_group_from_cache
            return
        if eval_cache_path is not None:
            logger.info(
                f"Creating MultiLevelDataset for evaluation from cached files in {rpath(eval_cache_path)}."
            )
            _cache_dir = Path(eval_cache_path)
            self.file_info_for_encoding_cache = (
                _cache_dir / EVAL_CACHE_FILE_NAMES["file_info_for_encoding"]
            )
            self.qrel_with_global_ids_cache = (
                _cache_dir / EVAL_CACHE_FILE_NAMES["global_qrel"]
            )
            self.qrel_with_local_ids_cache = (
                _cache_dir / EVAL_CACHE_FILE_NAMES["local_qrel"]
            )
            return

        if not isinstance(qrel_config, list):
            qrel_config = [qrel_config]

        # This is for logging purposes only. Do NOT use
        self._qrel_config = qrel_config

        _all_files = list()
        for conf in qrel_config:
            conf.ensure_list_of_correct_dtype()  # make sure everything is of type 'List'
            if conf.query_path is not None:
                if isinstance(conf.query_path, list):
                    _all_files.extend(conf.query_path)
                else:
                    _all_files.append(conf.query_path)

            if conf.corpus_path is not None:
                if isinstance(conf.corpus_path, list):
                    _all_files.extend(conf.corpus_path)
                else:
                    _all_files.append(conf.corpus_path)

        # Create a unique file id for each file path used in this dataset
        # This is used by MaterializedQRel instances to create globally unique query/passage IDs across files
        # which allows us to merge various files without concerns about duplicate IDs
        fname_to_id_map = file_utils.create_file_name_to_id_mapping(
            files=_all_files, id_length="shortest"
        )

        all_record_mappings = dict()
        for path in _all_files:
            _realpath = Path(path).resolve().absolute().resolve().as_posix()
            if _realpath not in all_record_mappings:
                all_record_mappings[_realpath] = containers.RowsByKeySingleSource(
                    filepath=path,
                    new_key_field="_id",
                    add_filepath_field=True,
                    num_proc=num_proc,
                )

        # Load the collections that make up this dataset.
        logger.info("Load all MaterializedQRel that make up this dataset.")
        self.qrel_collections = [
            MaterializedQRel(
                args=item,
                file_name_to_id=fname_to_id_map,
                record_mapping_collection=all_record_mappings,
                num_proc=num_proc,
            )
            for item in qrel_config
        ]

        all_qids = list(
            itertools.chain.from_iterable(
                [p.all_qids_global for p in self.qrel_collections]
            )
        )
        # ID of queries that are available in this dataset
        # Sort them for consistency across different runs
        self.all_qids = list(sorted(list(set(all_qids))))

    def update_metadata(self) -> None:
        """Updates the metadata for the dataset.

        It creates a new fingerprint and metadata dict for the dataset.
        """
        logger.debug("Creating dataset's fingerprint and info objects.")
        # The underlying data for each dataset is the set of MaterializedQRel collections
        # So its fingerprint is also the combined fingerprint of the MaterializedQRel collections that it contains.
        hasher = Hasher()
        for qrel_col in self.qrel_collections:
            hasher.update(qrel_col.fingerprint)

        # This is a workaround and not optimal. It should be changed in future.
        # There is some information in each MaterializedQRel that does not change
        # the state of the dataset and they are just there for ease of use and as
        # convenience tricks (e.g., MaterializedQRel.args.query_cache). Since changing them
        # does not change the state of the dataset, they ideally should not be included in dataset fingerprint.
        # However, if the user caches the dataset (with export_and_load_* methods) and then in a different run
        # they change these convenience fields and try to load the cache, the loaded dataset should contain the new values
        # for convenience fields.
        # In the final design, the convenience info should not be cached and instead their current value should be passed
        # to the cached datasets __init__ method when loading it in export_and_load_* methods.
        # Until then, we write and read them from cache and include them in the fingerprint so if they change,
        # the cache is invalidated and a new cache is created (which is not optimal.
        # All files in new cache will be the same except the convenience info files.)
        _encoding_path_info = self._get_encoding_path_info()
        for ftype_k in sorted(_encoding_path_info):
            for fpath_k in sorted(_encoding_path_info[ftype_k]):
                for field_k in sorted(_encoding_path_info[ftype_k][fpath_k]):
                    hasher.update(ftype_k)
                    hasher.update(fpath_k)
                    hasher.update(field_k)
                    hasher.update(_encoding_path_info[ftype_k][fpath_k][field_k])
        self._unique_fingerprint = hasher.hexdigest()
        self._info = {
            "fingerprint": self._unique_fingerprint,
            "data_args": self.args.to_dict(),
            "qrel_config": [qc.to_dict() for qc in self._qrel_config],
        }

    @property
    def fingerprint(self) -> str:
        """A unique fingerprint for the contents and output of this dataset.

        Datasets with the same fingerprint are backed by the same underlying data but do **NOT**
        necessarily generate the same samples. For example, using different sampling strategies
        or query and document formatting functions leads to different output from the same underlying
        data and thus the same fingerprint.

        This fingerprint is for internal operations only and you should not rely on it. And if you do,
        just use it to identify the underlying data (e.g., for caching and loading) and not
        the exact samples.

        datasets with the same fingerprint are backed by the same data sources.
        """
        if self._unique_fingerprint is None:
            self.update_metadata()
        return self._unique_fingerprint

    @property
    def info(self) -> Dict:
        if self._info is None or self._info["fingerprint"] != self.fingerprint:
            self.update_metadata()
        return self._info

    def set_index_lookup_storage_type(self, storage: str) -> None:
        """Select if the key to row index lookup table should be stored in memory or in memory-
        mapped lmdb dict."""
        for qc in self.qrel_collections:
            qc.set_index_lookup_storage_type(storage)

    def create_group_from_cache(self, index: int) -> Dict[str, Union[str, List[Dict]]]:
        """Loads one query and its related passages.

        Input and output is the same as :meth:`create_group_on_the_fly` but reads the data from
        cache file rather than creating it on-the-fly.
        """
        return self.train_records[index]

    def create_group_on_the_fly(self, index: int) -> Dict[str, Union[str, List[Dict]]]:
        """Loads one query and its related passages.

        The content of query/passages is also loaded (i.e., records are materialized.).

        If a passage exists in multiple collections, we use the data from the last collection
        in the self.qrel_collections list.

        Args:
            index (int): Index of the query to load

        Returns:

            A dict of the following format::

                {
                    'query_id': 'Unique ID of query across all files in this dataset'.
                    'query': 'query text',
                    'passages': [ # list of related passages for this query
                    {'_id': 'globally unique id of the passage', 'text': '...', 'title': '...'} # There could be additional fields in this dict, which should be ignored
                    ...,
                    {'_id': ....}
                    ]
                }
        """
        g_qid = self.all_qids[index]

        # Collect query/passage records in dicts from `_id` to `record` to avoid duplicate records
        query_rec = dict()
        doc_recs = dict()

        # Load related passages from all sources
        for qrel_col in self.qrel_collections:
            _q, _docs = qrel_col.get_related_recs(
                global_qid=g_qid, materialize=True, return_global_ids=True, strict=False
            )
            if _q is None or _docs is None:
                # this collection does not have entries for this specific query
                continue

            query_rec[_q["_id"]] = _q
            for doc in _docs:
                doc_recs[doc["_id"]] = doc

        # Make sure all collections return the same data for this query
        assert len(query_rec) == 1
        query_rec = list(query_rec.values())[0]
        doc_recs = list(doc_recs.values())

        group = {
            "query_id": query_rec["_id"],
            "query": query_rec["text"],
            "passages": doc_recs,
        }
        return group

    def __len__(self) -> int:
        if self.train_records is not None:
            return len(self.train_records)
        else:
            return len(self.all_qids)

    def __getitem__(
        self, index: int
    ) -> Dict[str, Union[str, List[str], List[Union[int, float]]]]:
        """Create a training example consisting of one query and a list of related passages.

        Returns:

            a dict with the following format::

                {
                    'query': formatted query of type `str`,
                    'passage': a list of formatted passages for the query.
                    'label': the relevancy level of each passage (i.e., score returned from qrel collections).
                }
        """
        group = self.create_group(index)
        if self._iter_over_groups:
            # we just need to return the training groups (and not training instances) without any processing
            return group
        query = group["query"]

        formatted_query = self.format_query(text=query, dataset=self.args.dataset_name)

        if self.args.group_size is None or self.args.group_size <= 0:
            # use all passages available
            passage_recs = group["passages"]
        else:
            # Sort the passage indices according to the given strategy
            # such that the index of the desired passages comes first
            # Also sort by '_id' field to make results consistent across different runs
            if self.args.passage_selection_strategy is None:
                curr_indices = list(range(len(group["passages"])))
            elif self.args.passage_selection_strategy == "random":
                _num_total = len(group["passages"])
                curr_indices = np.random.choice(
                    _num_total, [_num_total], replace=False
                ).tolist()
            elif self.args.passage_selection_strategy == "least_relevant":
                curr_indices = sorted(
                    range(len(group["passages"])),
                    key=lambda idx: (
                        group["passages"][idx]["score"],
                        group["passages"][idx]["_id"],
                    ),
                )
            elif self.args.passage_selection_strategy == "most_relevant":
                curr_indices = sorted(
                    range(len(group["passages"])),
                    key=lambda idx: (
                        group["passages"][idx]["score"],
                        group["passages"][idx]["_id"],
                    ),
                    reverse=True,
                )
            else:
                raise ValueError

            # choose the first self.args.group_size passages in the list
            # To also cover the case where slef.args.group_size > len(group['passages']),
            # we resample from the same passages again until reaching self.args.group_size
            chosen_idx = list(
                itertools.islice(itertools.cycle(curr_indices), self.args.group_size)
            )

            passage_recs = [group["passages"][i] for i in chosen_idx]

        formatted_passages = []
        passage_scores = list()
        for psg in passage_recs:
            formatted_passages.append(
                self.format_passage(
                    text=psg["text"],
                    title=psg["title"],
                    dataset=self.args.dataset_name,
                )
            )
            passage_scores.append(psg["score"])

        output = {
            "query": formatted_query,
            "passage": formatted_passages,
            "label": passage_scores,
        }
        return output

    def _get_encoding_path_info(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Return the info about file paths needed to create encoding datasets.

        Returns:

            A dict with two keys: ``query_files`` and ``corpus_files``. Each value is also a dict
            from a file path to a mapping from field name to field values::

                {
                    'query_files':
                        {
                            '/path/to/file': {
                                'path': ...,
                                '_id': ...,
                                'orig_cache_path': ...,
                            }
                        }
                }
        """
        if self.file_info_for_encoding_cache is not None:
            # It is a cached eval dataset. So just return the cached values for the output of this method
            with open(self.file_info_for_encoding_cache, "r") as f:
                path_info_dict = json.load(f)
            return path_info_dict

        # Use mappings from absolute resolved path to file info
        # to collect a list of data files without duplicates.
        query_files = dict()
        corpus_files = dict()
        for qrel_col in self.qrel_collections:
            for path, _cache in zip(
                qrel_col.args.query_path, qrel_col.args.query_cache
            ):
                _path_info = {
                    "path": path,
                    "_id": qrel_col.file_name_to_id[path],
                    "orig_cache_path": _cache,
                }
                realpath = Path(path).absolute().resolve().as_posix()
                if realpath in query_files:
                    assert query_files[realpath]["_id"] == _path_info["_id"]
                query_files[realpath] = _path_info

            for path, _cache in zip(
                qrel_col.args.corpus_path, qrel_col.args.corpus_cache
            ):
                _path_info = {
                    "path": path,
                    "_id": qrel_col.file_name_to_id[path],
                    "orig_cache_path": _cache,
                }
                realpath = Path(path).absolute().resolve().as_posix()
                if realpath in corpus_files:
                    assert corpus_files[realpath]["_id"] == _path_info["_id"]
                corpus_files[realpath] = _path_info
        return {"corpus_files": corpus_files, "query_files": query_files}

    def get_encoding_datasets(
        self,
        get_cache_path: Optional[
            Callable[[os.PathLike, str, Optional[os.PathLike]], Optional[os.PathLike]]
        ] = None,
        encoding_cache_pardir: Optional[os.PathLike] = None,
        **kwargs,
    ) -> Tuple[List[EncodingDataset], List[EncodingDataset]]:
        """Generates encoding datasets for query and corpuses used in this dataset.

        This is most useful for inference and evaluation. You can use the datasets in the
        output of this method to calculate the query/corpus embeddings and then similarity scores between them.
        You can also use :meth:`get_qrel_nested_dict` to get the groundtruth qrels as a nested dict.
        With the groundtruth qrels and the calculated similarity scores, you can compute the IR evaluation metrics for the model.

        You can optionally assign new cache path to the encoding datasets.
        If you do so, the cache path in qrel collection arguments is ignored.
        You can use the arguments to this function to dynamically set the cache path for encoding datasets.

        Args:
            get_cache_path (Optional[Callable]): A callable that generates the cache path for each encoding dataset.
                The callable should take three keyword arguments:

                * filepath (`os.PathLike`):  path to the input data file
                * file_id (`str`): globally unique _id for this file (see code of ``__init__`` function for more info)
                * orig_cache_path (`Optional[PathLike]`): the corresponding cache path for this filepath saved in ``MaterializedQRel.args``

                And it should return ``None`` or the filepath to the cache for this dataset
            encoding_cache_pardir (Optional[os.PathLike]): If

                * ``get_cache_path`` is not provided (i.e., is ``None``)
                * and ``encoding_cache_pardir`` is not ``None``
                * and ``orig_cache_path`` is a relative filepath that does not exist on disk

                then, we assume ``orig_cache_path`` is a relative filepath and use
                ``Path(encoding_cache_pardir, orig_cache_path)`` as the cache path for the encoding dataset.
            kwargs: keyword arguments passed to `EncodingDataset.__init__`

        Returns:
            A tuple. The first item is a list of encoding datasets for queries used in this dataset
            The second item is a list of encoding datasets for corpuses used in this dataset.
        """
        logger.info(
            "Create EncodingDataset from query and corpus files used in this dataset."
        )

        if get_cache_path is None:

            def get_cache_path(
                filepath: os.PathLike,
                file_id: str,
                orig_cache_path: Optional[os.PathLike] = None,
            ) -> Optional[os.PathLike]:
                """Create the cache path for encoding datasets.

                See docstring for `get_encoding_datasets()` for details.
                """

                if encoding_cache_pardir is None or orig_cache_path is None:
                    return orig_cache_path
                if Path(orig_cache_path).root != "" or Path(orig_cache_path).exists():
                    return orig_cache_path
                return Path(encoding_cache_pardir, orig_cache_path).as_posix()

        _encoding_path_info = self._get_encoding_path_info()
        query_files = _encoding_path_info["query_files"]
        corpus_files = _encoding_path_info["corpus_files"]

        # instantiate the encoding datasets
        query_datasets = list()
        for path_info in query_files.values():
            query_datasets.append(
                EncodingDataset(
                    data_args=self.args,
                    query_path=path_info["path"],
                    format_query=self.format_query,
                    global_id_suffix=path_info["_id"],
                    cache_path=get_cache_path(
                        filepath=path_info["path"],
                        orig_cache_path=path_info["orig_cache_path"],
                        file_id=path_info["_id"],
                    ),
                    **kwargs,
                )
            )

        corpus_datasets = list()
        for path_info in corpus_files.values():
            corpus_datasets.append(
                EncodingDataset(
                    data_args=self.args,
                    corpus_path=path_info["path"],
                    format_passage=self.format_passage,
                    global_id_suffix=path_info["_id"],
                    cache_path=get_cache_path(
                        filepath=path_info["path"],
                        orig_cache_path=path_info["orig_cache_path"],
                        file_id=path_info["_id"],
                    ),
                    **kwargs,
                )
            )
        return query_datasets, corpus_datasets

    def get_qrel_nested_dict(
        self, return_global_ids: bool = True
    ) -> Dict[str, Dict[str, Union[int, float]]]:
        """Collect the qrel triplets from all qrel collections into the nested dict format used by
        ``pytrec_eval``.

        Args:
            return_global_ids (bool): if true, use global IDs for queries and documents.

        Returns:
            a nested dict where ``dict[qid][docid]`` is the score between query ``qid`` and document ``docid`` in qrel files.
        """
        logger.info("Create nested qrel dict for MultiLevelDataset instance.")

        if return_global_ids:
            cache_file = self.qrel_with_global_ids_cache
        else:
            cache_file = self.qrel_with_local_ids_cache

        if cache_file is not None:
            # It is a cached eval dataset. So, return the cached values for the output of this method.
            with open(cache_file, "rb") as f:
                qrel_dump = pkl.load(f)
            return qrel_dump

        qrel_mapping = defaultdict(dict)
        for qrel_col in self.qrel_collections:
            _qrel_dict = qrel_col.get_qrel_nested_dict(
                return_global_ids=return_global_ids
            )
            for qid, qdata in tqdm(
                _qrel_dict.items(),
                desc="Append Qrel Dict from one QRelCol",
                disable=not logging_conf.is_debug(),
            ):
                for docid, docscore in qdata.items():
                    qrel_mapping[qid][docid] = int(docscore)
        qrel_mapping = dict(qrel_mapping)
        return qrel_mapping

    def export_and_load_train_cache(
        self,
        cache_file: Optional[os.PathLike] = None,
        cache_pardir: Optional[os.PathLike] = None,
        num_proc: Optional[int] = None,
        batch_size: int = 16,
    ):
        """Export the training groups to a cache file and load them into a new instance of
        ``MultiLevelDataset``.

        To reduce memory consumption, it generates all the training groups, write them
        into a json lines file and returns a new ``MultiLevelDataset`` instance from those cached records.

        To benefit from the reduced memory consumption, make sure you do not keep any references to the old dataset instance,
        so it can be garbage collected by the interpreter. You can do something like::

            dataset = MultiLevelDataset(...)
            dataset = dataset.export_and_load_train_cache()
            gc.collect() # if you want to force the interpreter to release the memory right away

        Args:
            cache_file: a json lines files to save the cached training groups. If ``None``, a unique cache
                file is created based on the dataset fingerprint.
            cache_pardir: the directory to save the cache file to. If provided, we create a
                a subdir in dis directory based on the dataset fingerprint and save the dataset cache
                in that subdir.
            num_proc: number of workers to use to generate the training groups.
            batch_size: read the training groups in batches of this size

        Returns:
            A new instance of ``MultiLevelDataset`` for training that is backed by the cached training groups.
        """
        logger.info(
            "Export MultiLevelDataset training groups to cache file and load them in a new instance."
        )
        if num_proc is None:
            num_proc = 0

        if cache_file is not None and cache_pardir is not None:
            msg = (
                "You can pass at most one of the 'cache_file' or 'cache_pardir' arguments."
                f" Got: 'cache_pardir': '{cache_pardir}' and 'cache_file': '{cache_file}'"
            )
            raise ValueError(msg)

        if cache_file is None:
            if cache_pardir is not None:
                cache_file = cache_manager.get_cache_dir(
                    cache_pardir=cache_pardir,
                    fingerprint=self.fingerprint,
                    metadata=self.info,
                )
            else:
                cache_file = cache_manager.get_cache_dir(
                    artifact_content="multilevel_dataset_train_cache",
                    artifact_type="final",
                    fingerprint=self.fingerprint,
                    metadata=self.info,
                )
            cache_file = Path(cache_file, "train_records.jsonl")
        else:
            cache_file = Path(cache_file)
        cache_file.parent.mkdir(exist_ok=True, parents=True)

        logger.debug("Waiting to acquire the cache file lock.")
        with file_utils.easyfilelock(cache_file.as_posix()):
            if not cache_file.exists():
                logger.debug("Dataset cache file does not exist. Start creating it.")
                # We don't want to save the exact output of the dataset since that depends on sampling strategy, etc.
                # Those operations must be done during training and do not have a noticeable impact memory consumption
                orig_iter_groups = self._iter_over_groups
                self._iter_over_groups = True

                # use torch data loaders parallelize and speedup the read operations
                data_loader = DataLoader(
                    self,
                    batch_size=batch_size,
                    collate_fn=lambda x: x,
                    num_workers=num_proc,
                    drop_last=False,
                    shuffle=False,
                )
                with file_utils.atomic_write(file=cache_file, root="parent") as tfile:
                    with JSONLinesWriter(tfile) as rec_writer:
                        for batch in tqdm(
                            data_loader, desc="Export train groups to jsonl file"
                        ):
                            rec_writer.add(batch)

                # revert the dataset to its original format/functionality
                self._iter_over_groups = orig_iter_groups
        logger.debug("Instantiate a new instance of the dataset from the cache files.")
        # do not forget to deepcopy attributes that are part of this instance (and not references to other objects)
        # If you do not do so, this instance will not get garbage collected, which makes the entire caching pointless
        return MultiLevelDataset(
            data_args=copy.deepcopy(self.args),
            format_query=self.format_query,
            format_passage=self.format_passage,
            train_cache_path=cache_file,
            num_proc=num_proc,
        )

    def export_and_load_eval_cache(
        self,
        cache_dir: Optional[os.PathLike] = None,
        cache_pardir: Optional[os.PathLike] = None,
    ):
        """Export the data required for evaluation to cache files and load them into a new instance
        of ``MultiLevelDataset``.

        To reduce memory consumption, it creates all the data that is required for evaluation
        and writes them into cache files and returns a new ``MultiLevelDataset`` instance from those cache files.

        To benefit from the reduced memory consumption, make sure you do not keep any references to the old dataset instance,
        so it can be garbage collected by the interpreter. You can do something like::

            dataset = MultiLevelDataset(...)
            dataset = dataset.export_and_load_eval_cache()
            gc.collect() # if you want to force the interpreter to release the memory right away

        Args:
            cache_dir: a directory where cache files should be saved. If ``None``, create a unique
                cache directory based on the dataset fingerprint.
            cache_pardir: the parent directory to save the cache file to. If provided, we create a
                a subdir in this directory based on the dataset fingerprint and save the dataset cache
                in that subdir.

        Returns:
            A new instance of ``MultiLevelDataset`` for evaluation that is backed by the cached files.
        """
        logger.info("Write MultiLevelDataset evaluation data to cache files.")

        if cache_dir is not None and cache_pardir is not None:
            msg = (
                "You can pass at most one of the 'cache_dir' or 'cache_pardir' arguments."
                f" Got: 'cache_pardir': '{cache_pardir}' and 'cache_dir': '{cache_dir}'"
            )
            raise ValueError(msg)

        if cache_dir is None:
            if cache_pardir is not None:
                cache_dir = cache_manager.get_cache_dir(
                    cache_pardir=cache_pardir,
                    fingerprint=self.fingerprint,
                    metadata=self.info,
                )
            else:
                cache_dir = cache_manager.get_cache_dir(
                    artifact_content="multilevel_dataset_eval_cache",
                    artifact_type="final",
                    fingerprint=self.fingerprint,
                    metadata=self.info,
                )
        else:
            cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True, parents=True)

        # we only need these three files for evaluation
        path_info_file = cache_dir / EVAL_CACHE_FILE_NAMES["file_info_for_encoding"]
        global_qrel_file = cache_dir / EVAL_CACHE_FILE_NAMES["global_qrel"]
        # local_qrel_file = cache_dir / EVAL_CACHE_FILE_NAMES["local_qrel"] # ignore this for the pre-release version

        # a successful caching requires all these files to be written without any corruption
        # so lock them all (i.e., count them as one giant super file)
        with (
            file_utils.easyfilelock(path_info_file),
            file_utils.easyfilelock(global_qrel_file),
            # file_utils.easyfilelock(local_qrel_file), # ignore this for the pre-release version
        ):
            # if any of the files does not exist, then the cache is not valid and should be overwritten
            if not (
                path_info_file.exists()
                and global_qrel_file.exists()
                # and local_qrel_file.exists() # ignore this for the pre-release version
            ):
                _encoding_path_info = self._get_encoding_path_info()
                _global_qrel = self.get_qrel_nested_dict(return_global_ids=True)
                # _local_qrel = self.get_qrel_nested_dict(return_global_ids=False)

                # do an atomic_write operation for all files at the same time so if
                # only one of them fails to write, all the rest will also fail
                with (
                    file_utils.atomic_write(
                        file=path_info_file, root="parent"
                    ) as path_f,
                    file_utils.atomic_write(
                        file=global_qrel_file, root="parent"
                    ) as gbl_f,
                    # file_utils.atomic_write(file=local_qrel_file, root="parent") as lcl_f, # ignore this for the pre-release version
                ):
                    with open(path_f, "w") as f:
                        json.dump(_encoding_path_info, f, indent=2)
                    with open(gbl_f, "wb") as f:
                        pkl.dump(_global_qrel, f)
                    # ignore this for the pre-release version
                    # with open(lcl_f, "wb") as f:
                    #     pkl.dump(_local_qrel, f)
        # do not forget to deepcopy attributes that are part of this instance (and not references to other objects)
        # If you do not do so, this instance will not get garbage collected, which makes the entire caching pointless
        return MultiLevelDataset(
            data_args=copy.deepcopy(self.args),
            format_query=self.format_query,
            format_passage=self.format_passage,
            eval_cache_path=cache_dir,
        )
