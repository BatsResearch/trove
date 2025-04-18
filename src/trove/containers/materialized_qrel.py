import itertools
import math
import os
import pickle as pkl
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import datasets
import numpy as np
from datasets.fingerprint import Hasher
from tqdm import tqdm

from .. import cache_manager, file_utils
from ..data import file_reader
from ..logging_utils import get_logger_with_config
from . import container_utils, rows_by_key
from .key_to_row_indices import KeyToRowIndices
from .materialized_qrel_config import MaterializedQRelConfig
from .rows_by_key import RowsByKeySingleSource

logger, logging_conf = get_logger_with_config("trove")


class MaterializedQRel:
    def __init__(
        self,
        args: MaterializedQRelConfig,
        file_name_to_id: Optional[Dict[os.PathLike, str]] = None,
        record_mapping_collection: Optional[Dict[str, RowsByKeySingleSource]] = None,
        num_proc: Optional[int] = None,
    ) -> None:
        """Represents a collection of query and documents and the relation between them.

        You can use this class to combine multiple query files with multiple corpus files and represent the relation between them with a combination of multiple qrel files.

        For the collection of queries:

            * Each query should have a unique ID
            * Each query should appear only once in each file
            * Each query should only appear in one file
            * In sum, each query ID should only appear once across all your files.

        At the moment, we do not impose these restrictions strictly as it is computationally expensive for large collections.
        You will get logically wrong results (or in consistent, at best) if your files do not follow these restrictions.

        The same is true for the collection of corpus files and the collection of qrel files.
        For qrel files, each record should be unique with respect to the combination of ``qid`` and ``docid``.
        I.e, each (``qid``, ``docid``) combination should only appear once across all your qrel files.


        If you do not specify a qrel file (i.e., ``args.qrel_path`` is empty) You can
        still use this class for further organization as a namespace container to
        hold a query and a corpus collection (or just one of them) without any
        known relation between them. E.g. hold a query and corpus collection for hard negative mining
        without knowing the relevance levels between the two. Another example is a ``MaterializedQRel`` instance
        that only has a collection of documents (without qrels or even queries). You can mix such an instance with other MaterializedQRel
        instances in a :py:class:`trove.data.ir_dataset_multilevel.MultiLevelDataset` instance to expand the document pool during nearest neighbor search
        without impacting the qrel triplets (without adding new qrel triplets).

        Args:
            args (MaterializedQRelConfig): Information about files that contain the raw data and how to process them.
            file_name_to_id (Optional[Dict[os.PathLike, str]]): A mapping from file name to a unique ID (of type ``str``) for that file.
                Although each ``qid`` and ``docid`` is unique in each instance of ``MaterializedQRel``,
                you might want to combine multiple instances of ``MaterializedQRel`` that could potentially assign the same ``qid`` or ``docid`` to different examples.
                To make that possible and uniquely identify each record across files, we can create a new ID by combining the original ID (which is unique in each file)
                with a suffix that uniquely identifies the file that contains the record with that ID.
                ``file_name_to_id`` is the mapping from file names to fild IDs that are used for this purposes.
                if not provided, we use create this mapping based on the hash of the file bytes.
                You can update the file IDs later, but that leads to repeating a lot of the computations.
            record_mapping_collection (Optional[Dict[str, RowsByKeySingleSource]]): instance of :py:class:`trove.data.ir_dataset_multilevel.RowsByKeySingleSource` for query and corpus files.
                If provided, use these instances instead of loading the data into new instances.
            num_proc (Optional[int]): passed to `datasets.Dataset.*` methods.
        """
        # A unique fingerprint for the contents and output of this collection
        # do not use this directly. Instead use 'self.fingerprint' attribute
        self._unique_fingerprint = None
        # some metadata about this container to be saved along with any cache files if necessary
        # Do not use this directly. Instead, use 'self.info' attribute
        self._info = None
        # parent directory where cache files should be saved. Do not use directly.
        # Instead, use self.cache_dir
        self._cache_dir = None

        args.ensure_list_of_correct_dtype()
        self.args = args

        if file_name_to_id is not None:
            # Since different paths might refer to the same file (e.g., symlinks),
            # use the resolved absolute path (similar to linux realpath command) to refer to files
            _fn2id_rp = {file_utils.realpath(k): v for k, v in file_name_to_id.items()}
            # Find the new file IDs for files used in this class.
            self.file_name_to_id = dict()
            for key in list(set(self.args.query_path + self.args.corpus_path)):
                self.file_name_to_id[key] = _fn2id_rp[file_utils.realpath(key)]
        else:
            # Since we are sure there are no duplicate files, we can
            # use the returned name_to_id mapping as self.file_name_to_id directly
            self.file_name_to_id = file_utils.create_file_name_to_id_mapping(
                files=list(set(self.args.query_path + self.args.corpus_path)),
                id_length="shortest",
            )
        # Make sure the file IDs are unique
        assert len(self.file_name_to_id) == len(
            set(list(self.file_name_to_id.values()))
        )

        # If provided, qrel triplets are limited to this subset of queries
        qid_subset = set()
        if self.args.query_subset_path is not None:
            if not isinstance(self.args.query_subset_path, list):
                self.args.query_subset_path = [self.args.query_subset_path]
            for path in self.args.query_subset_path:
                qid_subset.update(
                    set(file_reader.load_qids(filepath=path, num_proc=num_proc))
                )
        qid_subset = sorted(list(qid_subset))

        logger.debug("Read qrel files to create MaterializedQRel's qrel_ds.")
        # List of datasets.Dataset, each with (qid, docid, score) columns.
        qrel_ds = list()
        for path in self.args.qrel_path:
            _subqrel = file_reader.load_qrel(filepath=path, num_proc=num_proc)
            if len(qid_subset):
                _subidx_mapping = KeyToRowIndices(
                    dataset=_subqrel, id_field="qid", num_proc=num_proc
                )
                _subidx = list()
                for _sid in qid_subset:
                    _subidx.append(_subidx_mapping.get(_sid, []))
                _subidx = itertools.chain.from_iterable(_subidx)
                _subqrel = _subqrel.select(_subidx)
            qrel_ds.append(_subqrel)

        if len(qrel_ds) == 0:
            qrel_ds = None
        elif len(qrel_ds) == 1:
            qrel_ds = qrel_ds[0]
        else:
            qrel_ds = datasets.concatenate_datasets(qrel_ds)

        user_filter_fn = self.args.filter_fn
        if user_filter_fn is None and (
            self.args.min_score is not None or self.args.max_score is not None
        ):
            user_filter_fn = lambda ex: (
                (self.args.min_score is None) or (self.args.min_score <= ex["score"])
            ) and ((self.args.max_score is None) or (ex["score"] < self.args.max_score))

        if user_filter_fn is not None and qrel_ds is not None:
            logger.debug(f"Filtering MaterializedQRel's qrel_ds")
            # In batched .map, if a group ends up being empty, it will be removed, which is equivalent to filtering
            # Why not use .filter directly? with that we can either keep/filter a group
            # and not able to select a subset of a group
            qrel_ds = qrel_ds.map(
                container_utils.filter_fn_wrapper(user_filter_fn),
                remove_columns=qrel_ds.column_names,
                batched=True,
            )

        # Collection of all qrel triplets that could be easily filtered by 'qid'
        self.qrel_mapping = rows_by_key.RowsByKey(
            dataset=qrel_ds, key_field="qid", num_proc=num_proc
        )

        # Score transformation is applied to query-passage scores before returning them
        # If you define score transform function directly in the body of the __init__ method
        # then whenever you want to calculate the hash of the class (and thus hash of self.score_transform),
        # you pickle the entire closure of self.score_transform (which is __init__ local scope)
        # That is huge, takes a lot of resources and time. So define the function in a separate
        # method to avoid that issue
        self.score_transform = MaterializedQRel.create_score_transform(
            args_score_transform=self.args.score_transform
        )

        # see above comments for score transform for why we create the group_filter_fn
        # in a separate function
        # see docs for MaterializedQRelConfig for signature of group_filter_fn
        self.group_filter_fn = MaterializedQRel.create_group_filter_fn(
            args_group_filter_fn=self.args.group_filter_fn,
            group_top_k=self.args.group_top_k,
            group_bottom_k=self.args.group_bottom_k,
            group_first_k=self.args.group_first_k,
            group_random_k=self.args.group_random_k,
        )

        if record_mapping_collection is not None:
            # Since different paths might refer to the same file (e.g., symlinks),
            # use the resolved absolute path (similar to linux realpath command) to refer to files
            rec_map_col_realpath = {
                Path(k).resolve().absolute().resolve().as_posix(): v
                for k, v in record_mapping_collection.items()
            }
            corpus_rec_maps = list()
            for path in self.args.corpus_path:
                _realpath = Path(path).resolve().absolute().resolve().as_posix()
                rec_map = rec_map_col_realpath[_realpath]
                assert rec_map.key_field == "_id" and rec_map.add_filepath_field
                corpus_rec_maps.append(rec_map)

            self.corpus = rows_by_key.RowsByKey(
                record_mapping=corpus_rec_maps,
                allow_duplicates_per_id_lazy=False,
                num_proc=num_proc,
            )

            query_rec_maps = list()
            for path in self.args.query_path:
                _realpath = Path(path).resolve().absolute().resolve().as_posix()
                rec_map = rec_map_col_realpath[_realpath]
                assert rec_map.key_field == "_id" and rec_map.add_filepath_field
                query_rec_maps.append(rec_map)

            self.queries = rows_by_key.RowsByKey(
                record_mapping=query_rec_maps,
                allow_duplicates_per_id_lazy=False,
                num_proc=num_proc,
            )

        else:
            logger.info("Reading files to create MaterializedQRel.corpus")
            # All passage records accessible by docid
            self.corpus = rows_by_key.RowsByKey(
                filepath=self.args.corpus_path,
                new_key_field="_id",
                add_filepath_field=True,
                allow_duplicates_per_id_lazy=False,
                num_proc=num_proc,
            )

            logger.info("Reading files to create MaterializedQRel.queries")
            # All query records accessible by qid
            self.queries = rows_by_key.RowsByKey(
                filepath=self.args.query_path,
                new_key_field="_id",
                add_filepath_field=True,
                allow_duplicates_per_id_lazy=False,
                num_proc=num_proc,
            )
            logger.debug("Finished creating RowsByKey for corpus files.")

        # ID of all queries that we have related docs for
        self.all_qids_local = list(sorted(list(self.qrel_mapping.keys())))
        # If you are working with multiple instances of `MaterializedQRel` that contain conflicting IDs,
        # You should always use global IDs to interact with this class form outside.
        self.all_qids_global = self.get_global_qids()

    @staticmethod
    def create_score_transform(
        args_score_transform: Optional[
            Union[str, int, float, Callable[[Dict[str, Any]], Union[int, float]]]
        ]
    ) -> Callable[[Dict[str, Any]], Union[int, float]]:
        """Create score transform function from class arguments.

        This method should not be used when applying the ``score_transform`` function.
        Instead, you should call this in `__init__` to generate the score_transform callable
        and use that in the rest of the code.

        Args:
            args_score_transform: value of ``score_transform`` in ``MaterializedQRelConfig`` object.

        Returns:
            The score transform function that should be applied to each query/document score.
        """
        if args_score_transform is None:
            score_transform_fn = lambda x: x["score"]
        elif callable(args_score_transform):
            score_transform_fn = args_score_transform
        elif isinstance(args_score_transform, (int, float)):
            score_transform_fn = lambda _: args_score_transform
        elif isinstance(args_score_transform, str):
            if args_score_transform == "floor":
                score_transform_fn = lambda x: int(x["score"])
            elif args_score_transform == "ceil":
                score_transform_fn = lambda x: math.ceil(x["score"])
            else:
                raise ValueError
        else:
            raise TypeError
        return score_transform_fn

    @staticmethod
    def create_group_filter_fn(
        args_group_filter_fn: Optional[
            Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]
        ],
        group_top_k: Optional[int],
        group_bottom_k: Optional[int],
        group_first_k: Optional[int],
        group_random_k: Optional[int],
    ) -> Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]]:
        """Create ``group_filter_fn`` function from class arguments.

        You should call this in `__init__` to generate the ``group_filter_fn`` callable
        and use that in the rest of the code.

        Arguments are a subset of :py:class:`trove.containers.materialized_qrel_config.MaterializedQRelConfig` attributes. See its docstring
        for that class for details.
        """
        group_filter_fn = args_group_filter_fn
        if group_filter_fn is None:
            _non_empty_args = [
                group_top_k is not None,
                group_bottom_k is not None,
                group_first_k is not None,
                group_random_k is not None,
            ]
            assert sum(_non_empty_args) <= 1
            if group_first_k is not None:
                # choose first k documents in their original order
                group_filter_fn = lambda triplets: triplets[:group_first_k]
            elif group_bottom_k is not None:
                # choose k docs with lowest scores
                group_filter_fn = lambda triplets: sorted(
                    triplets, key=lambda x: x["score"]
                )[:group_bottom_k]
            elif group_top_k is not None:
                # choose k docs with largest scores
                group_filter_fn = lambda triplets: sorted(
                    triplets, reverse=True, key=lambda x: x["score"]
                )[:group_top_k]
            elif group_random_k is not None:
                group_filter_fn = lambda recs: (
                    [
                        recs[i]
                        for i in np.random.choice(
                            len(recs), [group_random_k], replace=False
                        )
                    ]
                    if len(recs) > group_random_k
                    else recs
                )
        return group_filter_fn

    def update_metadata(self) -> None:
        """Updates the metadata for the container.

        It creates a new fingerprint, ``cache_dir``, and metadata dict for the container.
        """
        # The combined hash of the following attributes uniquely identifies this container
        # A container with the same hash always produces the same output
        hasher = Hasher()
        for k in sorted(self.file_name_to_id):
            hasher.update(k)
            hasher.update(self.file_name_to_id[k])
        hasher.update(self.qrel_mapping.fingerprint)
        hasher.update(self.score_transform)
        if self.group_filter_fn is not None:
            hasher.update(self.group_filter_fn)
        hasher.update(self.corpus.fingerprint)
        hasher.update(self.queries.fingerprint)
        self._unique_fingerprint = hasher.hexdigest()
        self._info = {
            "fingerprint": self._unique_fingerprint,
            "args": self.args.to_dict(),
            "file_name_to_id": {
                Path(k).as_posix(): v for k, v in self.file_name_to_id.items()
            },
        }
        cache_dir = Path(
            cache_manager.get_cache_dir(
                artifact_content="materialized_qrel_cache",
                artifact_type="final",
                fingerprint=self.fingerprint,
                metadata=self.info,
            )
        )
        cache_dir.mkdir(exist_ok=True, parents=True)
        self._cache_dir = cache_dir.as_posix()

    @property
    def fingerprint(self) -> str:
        """A unique fingerprint for the contents and output of this container.

        Containers with the same fingerprint generate the same output.
        """
        if self._unique_fingerprint is None:
            self.update_metadata()
        return self._unique_fingerprint

    @property
    def info(self) -> Dict:
        if self._info is None or self._info["fingerprint"] != self.fingerprint:
            self.update_metadata()
        return self._info

    @property
    def cache_dir(self) -> Path:
        if self._cache_dir is None or self.info["fingerprint"] != self.fingerprint:
            self.update_metadata()
        return Path(self._cache_dir)

    def set_index_lookup_storage_type(self, storage: str) -> None:
        """Select if the key to row index lookup table should be stored in memory or in memory-
        mapped lmdb dict."""
        self.qrel_mapping.set_index_lookup_storage_type(storage)
        self.queries.set_index_lookup_storage_type(storage)
        self.corpus.set_index_lookup_storage_type(storage)

    def local_to_global_id(self, _id: str, _file: os.PathLike, **_) -> str:
        """Calculate the unique global ID across files for each local ID.

        The global ID is the local ID (i.e., original ID) plus the unique file ID appended at the end:
        ``global_id = original_id + '_' + unique_file_id``

        Args:
            _id (str): local id to generate a global id from
            _file (os.PathLike): the file that contains the record with this ID
            **_: Not used. Just to capture extra arguments

        Returns:
            The global ID (unique across files) corresponding the local ``_id``
        """
        return f"{_id}_{self.file_name_to_id[_file]}"

    def get_global_qids(self) -> List[str]:
        """Created global IDs for all queries."""
        logger.debug("Create global qids")
        with self.queries.meta_info_only():
            global_qids = [
                self.local_to_global_id(_id=qid, _file=self.queries[qid][0]["_file"])
                for qid in self.all_qids_local
            ]
        return global_qids

    def update_file_name_to_id(
        self, name_to_id_mapping: Dict[os.PathLike, str]
    ) -> None:
        """Assign new global IDs to files used in this class.

        Args:
            name_to_id_mapping (Dict[os.PathLike, str]): A mapping from filepath to its global ID.
        """
        # Since different paths might refer to the same file (e.g., symlinks),
        # use the resolved absolute path (similar to linux realpath command) to refer to files
        name_to_id_mapping_realpath = {
            Path(k).resolve().absolute().resolve().as_posix(): v
            for k, v in name_to_id_mapping.items()
        }

        # Find the new file IDs for files used in this class.
        new_mapping = dict()
        for key in self.file_name_to_id.keys():
            realpath_key = Path(key).resolve().absolute().resolve().as_posix()
            new_mapping[key] = name_to_id_mapping_realpath[realpath_key]

        if not all(
            [
                self.file_name_to_id[k] == new_mapping[k]
                for k in self.file_name_to_id.keys()
            ]
        ):
            # if IDs have changed, update the mapping and recalculate the global query IDs
            self.file_name_to_id = new_mapping
            if self.all_qids_global is not None:
                self.all_qids_global = self.get_global_qids()

    def _get_qrel_triplets(
        self, qid: str, default: Optional[Any] = None, strict: bool = True
    ) -> List[Dict[str, Union[str, float]]]:
        """Loads the qrel triplets for a given query.

        It takes care of necessary preprocessings like flattening grouped qrels and applying
        self.score_transform.

        Args:
            qid (str): ID of the query to load its related records.
            default (Optional[Any]): the default value for ``qrel_mapping.get()`` method.
            strict (bool): if true, use ``qrel_mapping[qid]``. Otherwise use ``qrel_mapping.get(qid, default)``

        Returns:
            A list of qrel triplets, where each item is a dict with three keys: ``qid``, ``docid``, and ``score``
        """

        if strict:
            qrel_recs = self.qrel_mapping[qid]
        else:
            qrel_recs = self.qrel_mapping.get(qid, default)
        qrel_recs = container_utils.flatten_grouped_triplets(qrel_recs)
        if self.group_filter_fn is not None:
            qrel_recs = self.group_filter_fn(qrel_recs)
        qrel_recs = [{**rec, "score": self.score_transform(rec)} for rec in qrel_recs]
        return qrel_recs

    def _get_related_recs(
        self,
        qid: str,
        materialize_full: bool = False,
        materialize_meta_info: bool = False,
    ) -> Tuple[Dict, List[Dict]]:
        """Get the query record and related document records for some qid.

        * ``query_rec`` is a dict with an ``_id`` field and if ``materialize_full`` or ``materialize_meta_info`` is ``True``, it also contains other fields
        such as ``text``, ``_file``, etc.

        * ``doc_recs`` is a list of all the documents related to ``qid`` based on the qrel files.
        I.e., list of documents that we have a corresponding (``qid``, ``docid``, ``score``) triplet for them.
        Each document is a dict with ``_id`` and ``score`` fields.If ``materialize_full`` or ``materialize_meta_info``
        is ``True``, it also includes additional fields like ``title``, ``text``, ``file``, etc.

        **The ``_id`` field of these records contain the local record IDs. Should not be used outside of this class.**

        Args:
            qid (str): ID of the query to load its related records.
            materialize_full (bool): If true, load the full content of query/document with
                all available fields (e.g. text) as well.
            materialize_meta_info (bool): If True, load the metadata fields for query/document (e.g., ``_file``)

        Returns:
            A tuple. First item is the corresponding query record and the second item
            is a list of related document records. If both ``materialize_full`` and ``materialize_meta_info``
            are ``False``, it only return the IDs and scores, without the main contentof query/document.
        """
        # corresponding triplets for qid
        qrel_recs = self._get_qrel_triplets(qid=qid, strict=True)
        if not len(qrel_recs):
            msg = f"There are no qrel triplets for query with qid: '{qid}'"
            raise ValueError(msg)

        if materialize_full or materialize_meta_info:
            with ExitStack() as stack:
                if not materialize_full:
                    stack.enter_context(self.queries.meta_info_only())
                    stack.enter_context(self.corpus.meta_info_only())

                # Extract the content of each query and document
                query_rec = self.queries[qid][0]
                doc_recs = list()
                for qrel_rec in qrel_recs:
                    rec = self.corpus[qrel_rec["docid"]][0]
                    doc_recs.append({**rec, "score": qrel_rec["score"]})
        else:
            # Only return _id and score
            query_rec = {"_id": qid}
            doc_recs = [
                {"_id": item["docid"], "score": item["score"]} for item in qrel_recs
            ]
        return query_rec, doc_recs

    def get_related_recs_for_local_qid(
        self,
        qid: str,
        materialize: bool = False,
        return_global_ids: bool = False,
        strict: bool = True,
    ) -> Tuple[Optional[Dict], Optional[List[Dict]]]:
        """Retrieve the related query and document records for a given ``qid``.

        See ``MaterializedQRel._get_related_recs`` for more details.

        Args:
            qid (str): See ``MaterializedQRel._get_related_recs`` for more details.
            materialize (bool): Even if this is false, the records could be materialized if ``return_global_ids`` is true.
                See ``self._get_related_recs()`` for more details.
            return_global_ids (bool): Whether to return global or local IDs in records
            strict (bool): Decides what to do if there are no corresponding qrel triplets for this qid.
                In such cases, if ``strict == False``, it returns a tuple of ``(None, None)``, and if ``strict == True`` , it raises an exception.

        Returns:
            A tuple of query record and related document records. See `self._get_related_recs()` for more details.
        """
        if not strict and qid not in self.all_qids_local:
            # There are no triplets for this query
            return None, None

        # Query and doc recs with local IDs
        query_rec, doc_recs = self._get_related_recs(
            qid=qid,
            materialize_full=materialize,
            materialize_meta_info=return_global_ids,
        )

        if return_global_ids:
            # Calculate and replace local IDs with global IDs
            if not materialize:
                # Create new records without additional fields if not asked for materialized records
                query_rec = {"_id": self.local_to_global_id(**query_rec)}
                for i in range(len(doc_recs)):
                    doc_recs[i] = {
                        "score": doc_recs[i]["score"],
                        "_id": self.local_to_global_id(**doc_recs[i]),
                    }
            else:
                # Only replace the local ids with global and keep other fields.
                query_rec["_id"] = self.local_to_global_id(**query_rec)
                query_rec.pop("_file", None)
                for i in range(len(doc_recs)):
                    doc_recs[i]["_id"] = self.local_to_global_id(**doc_recs[i])
                    doc_recs[i].pop("_file", None)
        elif materialize:
            # Remove the extra field from records
            query_rec.pop("_file", None)
            for i in range(len(doc_recs)):
                doc_recs[i].pop("_file", None)

        return query_rec, doc_recs

    def get_related_recs_for_global_qid(
        self,
        qid: str,
        materialize: bool = False,
        return_global_ids: bool = False,
        strict: bool = True,
    ) -> Tuple[Optional[Dict], Optional[List[Dict]]]:
        """Get related query and document records given the global id a query.

        See ``self.get_related_recs_for_local_qid()`` for details.

        Args:
            strict (bool): Decides what to do if there are no corresponding qrel triplets for this ``qid``.
                In such cases, if ``strict == False``, it returns a tuple of ``(None, None)``, and if ``strict == True``, it raises an exception.

        Returns:
            A tuple of query record and related document records.
            And ``(None, None)`` if there are no triplets for query with the given ID (when ``strict == False``).
            See ``self.get_related_recs_for_local_qid()`` for more details.
        """
        # Make sure to use the **global_qid** to check if we have any corresponding triplets for this query
        # do NOT use local_qid. local IDs across MaterializedQRel instances are not necessarily unique
        # and this method is intended to be called by other classes that work with multiple instances of MaterializedQRel
        if qid not in self.all_qids_global:
            # We do not have any records for this query
            if strict:
                msg = f"There are no qrel triplets for query with global qid '{qid}'"
                raise ValueError(msg)
            else:
                return None, None
        qid = qid.rsplit("_", maxsplit=1)[0]
        return self.get_related_recs_for_local_qid(
            qid=qid,
            materialize=materialize,
            return_global_ids=return_global_ids,
            strict=strict,  # no impact. But, pass the logic along for consistency, making further development easier
        )

    def get_related_recs(
        self,
        global_qid: Optional[str] = None,
        local_qid: Optional[str] = None,
        materialize: bool = False,
        return_global_ids: bool = True,
        strict: bool = True,
    ) -> Tuple[Optional[Dict], Optional[List[Dict]]]:
        """Get the related records for some query.

        If ``strict==False`` and there are no triplets for query with the given ID, it returns ``(None, None)``.
        See ``MaterializedQRel.get_related_recs_for_local_qid()`` for more details.

        Args:
            global_qid (Optional[str]): global ID of the query. Takes precedence over ``local_qid`` if provided.
            local_qid (Optional[str]): local ID of the query. It is ignored if ``global_qid`` is provided.
        """
        if (global_qid is None) == (local_qid is None):
            msg = "You should provide exactly one of the 'global_qid' or 'local_qid' arguments."
            raise ValueError(msg)

        if global_qid is not None:
            return self.get_related_recs_for_global_qid(
                qid=global_qid,
                materialize=materialize,
                return_global_ids=return_global_ids,
                strict=strict,
            )
        else:
            return self.get_related_recs_for_local_qid(
                qid=local_qid,
                materialize=materialize,
                return_global_ids=return_global_ids,
                strict=strict,
            )

    def get_qrel_nested_dict(
        self, return_global_ids: bool = False
    ) -> Dict[str, Dict[str, Union[int, float]]]:
        """Converts the qrel triplets to the nested dict format used by ``pytrec_eval``.

        Args:
            return_global_ids (bool): if true, use global ids for queries and documents.

        Returns:
            a nested dict where dict[qid][docid] is the score between query ``qid`` and document ``docid``.
        """
        logger.debug("Create nested qrel dict for MaterializedQRel.")
        # choose the correct cache file
        if return_global_ids:
            cache_file = self.cache_dir / "nested_qrel_with_global_ids.pkl"
        else:
            cache_file = self.cache_dir / "nested_qrel_with_local_ids.pkl"

        qrel_mapping = None
        with file_utils.easyfilelock(cache_file.as_posix()):
            if not cache_file.exists():
                qrel_mapping = defaultdict(dict)
                with self.queries.meta_info_only(), self.corpus.meta_info_only():
                    for qid in tqdm(
                        self.all_qids_local,
                        desc="Make Qrel Dict",
                        disable=not logging_conf.is_debug(),
                    ):
                        rel_docs = self._get_qrel_triplets(qid=qid, strict=True)
                        if return_global_ids:
                            rec_qid = self.local_to_global_id(**self.queries[qid][0])
                        else:
                            rec_qid = qid
                        for doc in rel_docs:
                            if return_global_ids:
                                docid = self.local_to_global_id(
                                    **self.corpus[doc["docid"]][0]
                                )
                            else:
                                docid = doc["docid"]
                            qrel_mapping[rec_qid][docid] = doc["score"]
                qrel_mapping = dict(qrel_mapping)
                with file_utils.atomic_write(file=cache_file, root="parent") as tfile:
                    with open(tfile, "wb") as f:
                        pkl.dump(qrel_mapping, f)

        if qrel_mapping is None:
            with open(cache_file, "rb") as f:
                qrel_mapping = pkl.load(f)
        return qrel_mapping
