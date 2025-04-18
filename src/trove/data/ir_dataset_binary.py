import copy
import itertools
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import datasets
from datasets.fingerprint import Hasher
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Trainer

from .. import cache_manager, containers, file_utils
from ..containers import MaterializedQRel, MaterializedQRelConfig
from ..file_utils import JSONLinesWriter
from ..logging_utils import get_logger_with_config, rpath
from .data_args import DataArguments

logger, logging_conf = get_logger_with_config("trove")


class BinaryDataset(Dataset):
    def __init__(
        self,
        data_args: DataArguments,
        format_query: Callable[[str, Optional[str]], str],
        format_passage: Callable[[str, str, Optional[str]], str],
        positive_configs: Optional[
            Union[MaterializedQRelConfig, List[MaterializedQRelConfig]]
        ] = None,
        negative_configs: Optional[
            Union[MaterializedQRelConfig, List[MaterializedQRelConfig]]
        ] = None,
        train_cache_path: Optional[os.PathLike] = None,
        data_args_overrides: Optional[Dict[str, Any]] = None,
        trainer: Optional[Trainer] = None,
        outside_trainer: bool = False,
        num_proc: Optional[int] = None,
    ) -> None:
        """IR training dataset with only two levels of relevance (i.e., only positive and negative
        passages).

        * collections of positive (negative) passages are created from ``positive_configs`` (``negative_configs``) argument.
        * passages that appear in either of the positive or negative collections will be included in the resulting dataset.

        Args:
            data_args (DataArguments): general arguments for loading and processing the data
            format_query (Callable[[str, Optional[str]], str]): A callable that takes the query
                text and optionally the dataset name and returns the formatted query text for the modle.
            format_passage (Callable[[str, str, Optional[str]], str]): A callable that takes
                the passage text and title and dataset name and returns the formatted passage text for the model.
            positive_configs (Optional[Union[MaterializedQRelConfig, List[MaterializedQRelConfig]]]):
                Config for one or multiple collections of queries, documents, and the relation between them.
                The passages from these collections are used as positives.
            negative_configs (Optional[Union[MaterializedQRelConfig, List[MaterializedQRelConfig]]]):
                Config for one or multiple collections of queries, documents, and the relation between them.
                The passages from these collections are used as negatives.
            train_cache_path (Optional[os.PathLike]): **DO NOT USE**. For internal operations only and not stable.
                If given, create a dataset only for training from this cache file.
                This is much more memory efficient compared to creating the dataset on-the-fly.
                You should use :meth:`export_and_load_train_cache` method to take advantage of this.
            data_args_overrides (Optional[Dict[str, Any]]): A mapping from a subset of ``DataArguments`` attribute names to
                their new values. These key values override the corresponding attributes in ``data_args`` argument.
                It is useful if you want to create multiple datasets from the same `DataArguments` instance
                but make small changes for each dataset without creating new ``DataArguments`` instances.
            trainer (Optional[Trainer]): An instance of the ``transformers.Trainer`` class.
                The random seed and epoch from trainer instance are used to sample positive and negative documents.
            outside_trainer (bool): If true, do not use trainer instance and set seed and epoch both to zero.
                Useful for debugging without a trainer instance.
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
        # self.create_group() which just returns the related docs) for each query rather than a training sample.
        # this is useful for iterating over the underlying data for each dataset instance.
        self._iter_over_groups = False

        if (positive_configs is None) != (negative_configs is None):
            msg = (
                "You should pass either both or none of the 'positive_configs' and 'negative_configs' arguments."
                f" Got: 'positive_configs': '{positive_configs}' and 'negative_configs': '{negative_configs}'"
            )
            raise ValueError(msg)

        if (positive_configs is not None or negative_configs is not None) == (
            train_cache_path is not None
        ):
            msg = (
                "You should pass exactly one of the ('positive_configs', 'negative_configs') pair or 'train_cache_path' arguments."
                f" Got: 'positive_configs': '{positive_configs}', 'negative_configs': '{negative_configs}', and 'train_cache_path': '{train_cache_path}'"
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

        self.format_query = format_query
        self.format_passage = format_passage

        self.outside_trainer = outside_trainer
        self.trainer = trainer

        # by default, the training groups should be created on-the-fly
        self.create_group = self.create_group_on_the_fly

        self.train_records = None
        if train_cache_path is not None:
            logger.info(
                f"Creating BinaryDataset from cached training groups in {rpath(train_cache_path)}."
            )
            if Path(train_cache_path).suffix != ".jsonl":
                msg = f"Only supported format for 'train_cache_path' is json lines. Got: {train_cache_path}"
                raise ValueError(msg)
            self.train_records = datasets.load_dataset(
                "json", data_files=Path(train_cache_path).as_posix(), split="train"
            )
            self.create_group = self.create_group_from_cache
            return

        # Make sure everything is of type 'list'
        if not isinstance(positive_configs, list):
            positive_configs = [positive_configs]
        if not isinstance(negative_configs, list):
            negative_configs = [negative_configs]

        # This is for logging purposes only. Do NOT use
        self._positive_configs = positive_configs
        self._negative_configs = negative_configs

        _all_files = list()
        for conf in positive_configs + negative_configs:
            conf.ensure_list_of_correct_dtype()
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

        # Create the necessary qrel collections for both positive and negative documents.
        logger.info("Load MaterializedQRel for positive passages.")
        self.positive_passages = [
            MaterializedQRel(
                args=item,
                file_name_to_id=fname_to_id_map,
                record_mapping_collection=all_record_mappings,
                num_proc=num_proc,
            )
            for item in positive_configs
        ]
        logger.info("Load MaterializedQRel for negative passages.")
        self.negative_passages = [
            MaterializedQRel(
                args=item,
                file_name_to_id=fname_to_id_map,
                record_mapping_collection=all_record_mappings,
                num_proc=num_proc,
            )
            for item in negative_configs
        ]

        qids_with_pos = list(
            itertools.chain.from_iterable(
                [p.all_qids_global for p in self.positive_passages]
            )
        )
        qids_with_neg = list(
            itertools.chain.from_iterable(
                [p.all_qids_global for p in self.negative_passages]
            )
        )

        # Queries that have both positive and negative passages and thus can be used by this dataset
        # Sort them for consistency across different runs
        self.all_qids = list(
            sorted(list(set(qids_with_pos).intersection(set(qids_with_neg))))
        )

    def update_metadata(self) -> None:
        """Updates the metadata for the dataset.

        It creates a new fingerprint and metadata dict for the dataset.
        """
        # The underlying data for each dataset is the set of MaterializedQRel collections
        # So its fingerprint is also the combined fingerprint of the MaterializedQRel collections that it contains.
        hasher = Hasher()
        for qc in self.negative_passages:
            hasher.update(qc.fingerprint)
        for qc in self.positive_passages:
            hasher.update(qc.fingerprint)
        self._unique_fingerprint = hasher.hexdigest()
        self._info = {
            "fingerprint": self._unique_fingerprint,
            "data_args": self.args.to_dict(),
            "positive_configs": [qc.to_dict() for qc in self._positive_configs],
            "negative_configs": [qc.to_dict() for qc in self._negative_configs],
        }

    @property
    def fingerprint(self) -> str:
        """Calculates a unique fingerprint for the contents and output of this dataset.

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
        for qc in self.negative_passages:
            qc.set_index_lookup_storage_type(storage)
        for qc in self.positive_passages:
            qc.set_index_lookup_storage_type(storage)

    def create_group_from_cache(self, index: int) -> Dict[str, Union[str, List[Dict]]]:
        """Loads one query and the related negative and positive passages.

        Input and output is the same as :meth:`create_group_on_the_fly` but reads the data from
        cache file rather than creating it on-the-fly.
        """
        return self.train_records[index]

    def create_group_on_the_fly(self, index: int) -> Dict[str, Union[str, List[Dict]]]:
        """Loads one query and the related negative and positive passages.

        The content of query/passages is also loaded (i.e., records are materialized.).
        The return format is based on `tevatron <https://github.com/texttron/tevatron>`_ .

        Args:
            index (int): Index of the query to load

        Returns:

            A dict of the following format::

                {
                    'query_id': 'Unique ID of query across all files in this dataset'.
                    'query': 'query text',
                    'positive_passages': [ # list of positive documents for this query
                    {'_id': 'globally unique id of the passage', 'text': '...', 'title': '...'}, # There could be additional fields in this dict, which should be ignored
                    ...,
                    {'_id': ....}
                    ],
                    'negative_passages': [ # list of negative documents for this query
                    {'_id': ....}, # The same data structure and field names as positive documents.
                    ...,
                    {'_id': ....}
                    ]

                }
        """
        qid = self.all_qids[index]

        # Collect query/passage records in dicts from `_id` to `record` to avoid duplicate records
        q_recs = dict()
        pos_recs = dict()
        neg_recs = dict()

        # Load negative passages from different sources
        for neg_col in self.negative_passages:
            query_rec, doc_recs = neg_col.get_related_recs(
                global_qid=qid, materialize=True, return_global_ids=True, strict=False
            )
            if query_rec is None or doc_recs is None:
                # this collection does not have entries for this specific query
                continue
            q_recs[query_rec["_id"]] = query_rec
            for d_rec in doc_recs:
                neg_recs[d_rec["_id"]] = d_rec

        # Load positive passages from different sources
        for pos_col in self.positive_passages:
            query_rec, doc_recs = pos_col.get_related_recs(
                global_qid=qid, materialize=True, return_global_ids=True, strict=False
            )
            if query_rec is None or doc_recs is None:
                # this collection does not have entries for this specific query
                continue
            q_recs[query_rec["_id"]] = query_rec
            for d_rec in doc_recs:
                pos_recs[d_rec["_id"]] = d_rec

        # Make sure all collections return the same data for this query
        assert len(q_recs) == 1
        q_rec = next(iter(q_recs.values()))

        if (
            len(set(list(pos_recs.keys())).intersection(set(list(neg_recs.keys()))))
            != 0
        ):
            msg = "Some documents are used as both positive and negatives for the same query."
            raise RuntimeError(msg)

        pos_docs = list(pos_recs.values())
        neg_docs = list(neg_recs.values())

        group = {
            "query_id": q_rec["_id"],
            "query": q_rec["text"],
            "positive_passages": pos_docs,
            "negative_passages": neg_docs,
        }

        return group

    def __len__(self) -> int:
        if self.train_records is not None:
            return len(self.train_records)
        else:
            return len(self.all_qids)

    def set_trainer(self, trainer: Trainer) -> None:
        """Set the trainer attribute."""
        if self.trainer is None:
            self.trainer = trainer

    def epoch_and_seed(self) -> Tuple[int, Union[float, int]]:
        """If trainer instance is available, load seed and current epoch from trainer."""
        if self.outside_trainer:
            epoch = 0
            seed = 0
        else:
            epoch = int(self.trainer.state.epoch)
            seed = self.trainer.args.seed
        return epoch, seed

    def __getitem__(
        self, index: int
    ) -> Dict[str, Union[str, List[str], List[Union[int, float]]]]:
        """Create a training example consisting of one query, one positive passage and multiple
        negative passages.

        taken from `here <https://github.com/texttron/tevatron/blob/7d298b428234f1c1065e98244827824753361815/src/tevatron/retriever/dataset.py#L41>`_
        With minor changes to make it work with this class.

        Returns:

            a dict with the following format::

                {
                    'query': formatted query of type `str`,
                    'passage': a list of formatted passages for the query.
                        The positive passage is the first element of the list and the rest are negatives.
                    'labels': the relevancy level of each passage (i.e., score).
                        Positive and negative passages are given scores of 1 and 0, respectively.
                }
        """
        group = self.create_group(index)
        if self._iter_over_groups:
            # we just need to return the training groups (and not training instances) without any processing
            return group
        epoch, trainer_seed = self.epoch_and_seed()

        _hashed_seed = hash(index + trainer_seed)

        query = group["query"]
        group_positives = group["positive_passages"]
        group_negatives = group["negative_passages"]

        formatted_query = self.format_query(text=query, dataset=self.args.dataset_name)
        formatted_passages = []
        passage_scores = list()

        if self.args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]

        formatted_passages.append(
            self.format_passage(
                text=pos_psg["text"],
                title=pos_psg["title"],
                dataset=self.args.dataset_name,
            )
        )
        passage_scores.append(1)

        negative_size = self.args.group_size - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.args.group_size == 1:
            negs = []
        elif self.args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset : _offset + negative_size]

        for neg_psg in negs:
            formatted_passages.append(
                self.format_passage(
                    text=neg_psg["text"],
                    title=neg_psg["title"],
                    dataset=self.args.dataset_name,
                )
            )
            passage_scores.append(0)

        output = {
            "query": formatted_query,
            "passage": formatted_passages,
            "label": passage_scores,
        }
        return output

    def export_and_load_train_cache(
        self,
        cache_file: Optional[os.PathLike] = None,
        cache_pardir: Optional[os.PathLike] = None,
        num_proc: Optional[int] = None,
        batch_size: int = 16,
    ):
        """Export the training groups to a cache file and load them into a new instance of
        BinaryDataset.

        To reduce memory consumption, it generates all the training groups, write them
        into a json lines file and returns a new ``BinaryDataset`` instance from those cached records.
        To benefit from the reduced memory consumption, make sure you do not keep any references to the old dataset instance,
        so it can be garbage collected by the interpreter.

        You can do something like::

            dataset = BinaryDataset(...)
            dataset = dataset.export_and_load_train_cache()
            gc.collect() # if you want to force the interpreter to release the memory right away

        Args:
            cache_file: a json lines files to save the cached training groups. If ``None``, a unique cache
                file is created based on the dataset fingerprint.
            cache_pardir: the parent directory to save the cache file to. If provided, we create a
                a subdir in this directory based on the dataset fingerprint and save the dataset cache
                in that subdir.
            num_proc: number of workers to use to generate the training groups.
            batch_size: read the training groups in batches of the given size

        Returns:
            A new instance of ``BinaryDataset`` for training that is backed by the cached training groups.
        """
        logger.info("Export BinaryDataset training groups to cache file.")
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
                    artifact_content="binary_dataset_train_cache",
                    artifact_type="final",
                    fingerprint=self.fingerprint,
                    metadata=self.info,
                )
            cache_file = Path(cache_file, "train_records.jsonl")
        else:
            cache_file = Path(cache_file)
        cache_file.parent.mkdir(exist_ok=True, parents=True)

        with file_utils.easyfilelock(cache_file.as_posix()):
            if not cache_file.exists():
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
        # do not forget to deepcopy attributes that are part of this instance (and not references to other objects)
        # If you do not do so, this instance will not get garbage collected, which makes the entire caching pointless
        return BinaryDataset(
            data_args=copy.deepcopy(self.args),
            format_query=self.format_query,
            format_passage=self.format_passage,
            train_cache_path=cache_file,
            trainer=self.trainer,
            outside_trainer=self.outside_trainer,
            num_proc=num_proc,
        )
