import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import huggingface_hub as hf_hub


@dataclass
class MaterializedQRelConfig:
    """Information about a collection of queries, documents, and (optionally) the relation between
    them.

    You can use both path to local files and remote HF hub fsspec URIs for ``qrel_path``, ``corpus_path``, ``query_path``, and ``query_subset_path``.
    HF hub URIs must start with ``hf://``. See `HF hub documentation <https://huggingface.co/docs/huggingface_hub/en/guides/hf_file_system>`_ for the exact structure.

    If you do not set the value of ``qrel_path`` or set it to an empty list, ``MaterializedQRel`` will
    be a namespace container without any information about the relation between queries and
    documents. See :py:class:`trove.containers.materialized_qrel.MaterializedQRel` docstring for details.
    """

    qrel_path: Optional[Union[os.PathLike, List[os.PathLike]]] = None
    """One or multiple files that contain triplets of ``('qid', 'docid', 'score')``.
    The files do not need to explicitely contain such triplets. We can also infer them from other types of data.
    Look at ``file_reader.load_qrel()`` for supported files.
    """
    corpus_path: Optional[Union[os.PathLike, List[os.PathLike]]] = None
    """One or multiple files that contain the passage text and optionally titles.
    """
    query_path: Optional[Union[os.PathLike, List[os.PathLike]]] = None
    """One or multiple files that contain the query texts.
    """
    corpus_cache: Optional[Union[os.PathLike, List[os.PathLike]]] = None
    """(Not directly used) A Corresponding cache file name for each of the ``corpus_path`` to read/write the resulting embedding vectors.
    We do not directly use this. We just save it as part of the ``MaterializedQRel.args`` that you can use later.
    For example, you can use this to store a unique relative filepath for each of the corpus files.
    Then, during runtime, calculate a parent directory (e.g., based on embedding model name, etc.) and combine it with the relative filepath to get the complete path to cache files.
    """
    query_cache: Optional[Union[os.PathLike, List[os.PathLike]]] = None
    """(Not directly used) A Corresponding cache file name for each of the ``query_path`` to read/write the resulting embedding vectors.
    See docstring for ``corpus_cache`` for details.
    """
    query_subset_path: Optional[Union[os.PathLike, List[os.PathLike]]] = None
    """One or multiple files that it is possible to read a list of query IDs from.
    The available qrel triplets are limited to these queries. See ``file_reader.load_qids()`` for the supported files.
    """
    min_score: Optional[Union[int, float]] = None
    """If provided, filter the qrel triplets and only keep ones with ``min_score <= score`` (Endpoint is included in the interval)
    """
    max_score: Optional[Union[int, float]] = None
    """If provided, filter the qrel triplets and only keep ones with ``score < max_score`` (Endpoint is NOT included in the interval)
    """
    filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    """A callable used for filtering qrel triplets. If provided, ``min_score`` and ``max_score`` are ignored.
    `filter_fn` should take a dict (content of the qrel triplet with ``qid``, ``docid``, and ``score`` keys) as input and return a boolean as output.
    It is used like ``datasets.Dataset.filter(filter_fn, ...)``. I.e., keep the record if ``filter_fn`` returns ``True``.
    """
    group_top_k: Optional[int] = None
    """If given, filter the available documents for each query and only choose the ``group_top_k`` documents with the highest score for each query."""
    group_bottom_k: Optional[int] = None
    """If given, filter the available documents for each query and only choose the ``group_bottom_k`` documents with the lowest score for each query."""
    group_first_k: Optional[int] = None
    """If given, filter the available documents for each query and only keep the first ``group_first_k`` documents (in their original ordering) for each query."""
    group_random_k: Optional[int] = None
    """If given, filter the available documents for each query and choose ``group_random_k`` documents randomly for each query.
    Return all documents if number of available documents per query is smaller than ``group_random_k``.
    """
    group_filter_fn: Optional[
        Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]
    ] = None
    """A callable used to filter the qrel triplets for each query. If given, it overrides ``group_first_k``, ``group_top_k``, ``group_bottom_k``, and ``group_random_k``.
    There are several differences between ``group_filter_fn`` and ``filter_fn``.

        * ``filter_fn`` is used in the __init__ function to filter all the triplets for all queries and get the collection of available qrel triplets.
          But ``group_filter_fn`` is called whenever you attempt to get a list of available triplets for some query (i.e., every time you call methods like ``get_related_recs_for_*``).
          Unlike ``filter_fn``, results of ``group_filter_fn`` are not cached.

        * ``filter_fn`` operates on individual qrel triplets. But, ``group_filter_fn`` operates on the list of all available qrel triplets for some query.

    ``group_filter_fn`` must be a callable that takes one positional argument. The argument is a list of dict objects.
    Each dict object is a qrel triplet for the query. The dict object contains keys ``qid``, ``docid``, ``score``, and potentially other keys.
    The input list contains all the available qrel triplets for this query (the list could be empty).
    This callable should return an output with the same format as its input (i.e., a list of dicts).
    The behavior of this class is unknown if the callable receives a non-empty list but returns an empty list.
    If given, this callable is called before calling ``score_transform``.
    This argument is useful for filtering documents based on other documents available for each query.
    For example, to only keep the N most similar items for each query.
    """
    score_transform: Optional[
        Union[str, int, float, Callable[[Dict[str, Any]], Union[int, float]]]
    ] = None
    """A transformation applied to scores at the very last step right before returning them.
    Acceptable types for ``score_transform`` are:

    * ``None`` : return the scores as is
    * ``callable`` : it should take a dict (content of the qrel triplet with ``qid``, ``docid``, and ``score`` keys) as input and return the transformed score as output. it will be used like ``new_score = score_transform(rec)``
    * ``Union[int, float]`` : This value is used as the score for all qrel triplets. I.e., score is a constant for all query-document pairs
    * ``str`` : A predefined behavior. At the moment ``floor`` and ``ceil`` are valid behaviors. ``floor`` and ``ceil`` return ``int(triplet['score'])`` and ``math.ceil(triplet['score'])``, respectively
    """

    def __post_init__(self):
        self.ensure_list_of_correct_dtype()
        _non_empty_args = [
            k is not None
            for k in [
                self.group_top_k,
                self.group_bottom_k,
                self.group_first_k,
                self.group_random_k,
            ]
        ]
        if sum(_non_empty_args) > 1:
            msg = (
                "You should choose only one of the 'group_top_k', 'group_bottom_k', 'group_first_k', 'group_random_k' arguments."
                f" Got 'group_top_k': '{self.group_top_k}', 'group_bottom_k': '{self.group_bottom_k}', 'group_first_k': '{self.group_first_k}', and 'group_random_k': '{self.group_random_k}'"
            )
            raise ValueError(msg)

    def ensure_list_of_correct_dtype(self):
        """Ensure everything of type ``List[str]``."""
        # These three attrs must be lists. So, if they are 'None', that means an empty list
        if self.qrel_path is None:
            self.qrel_path = []
        if self.query_path is None:
            self.query_path = []
        if self.corpus_path is None:
            self.corpus_path = []

        # For consistency, make sure everything is of type `list` even if it is a single item
        if not isinstance(self.qrel_path, list):
            self.qrel_path = [self.qrel_path]

        # If query_cache is None and we have multiple query_path,
        # make sure there is one 'None' for each file
        if self.query_cache is None and isinstance(self.query_path, list):
            self.query_cache = [None] * len(self.query_path)

        # There should be one query_cache for each query_path
        assert isinstance(self.query_path, list) == isinstance(self.query_cache, list)

        # For consistency, make sure everything is a of type `list`
        if not isinstance(self.query_path, list):
            self.query_path = [self.query_path]
            self.query_cache = [self.query_cache]

        # There is a one-to-one correspondence between query_path and query_cache
        assert len(self.query_path) == len(self.query_cache)

        # Make sure everything is of type `str`
        self.query_path = [str(p) for p in self.query_path]
        self.query_cache = [None if p is None else str(p) for p in self.query_cache]

        # Repeat the same thing for corpus files
        if self.corpus_cache is None and isinstance(self.corpus_path, list):
            self.corpus_cache = [None] * len(self.corpus_path)
        assert isinstance(self.corpus_path, list) == isinstance(self.corpus_cache, list)
        if not isinstance(self.corpus_path, list):
            self.corpus_path = [self.corpus_path]
            self.corpus_cache = [self.corpus_cache]

        # There is a one-to-one correspondence between corpus_path and corpus_cache
        assert len(self.corpus_path) == len(self.corpus_cache)

        self.corpus_path = [str(p) for p in self.corpus_path]
        self.corpus_cache = [None if p is None else str(p) for p in self.corpus_cache]

        if self.query_subset_path is not None:
            if not isinstance(self.query_subset_path, list):
                self.query_subset_path = [self.query_subset_path]
            self.query_subset_path = list(sorted(self.query_subset_path))

        self.qrel_path = list(sorted(self.qrel_path))

        corpus_sorted_idx = sorted(
            range(len(self.corpus_path)), key=lambda i: self.corpus_path[i]
        )
        self.corpus_path = [self.corpus_path[i] for i in corpus_sorted_idx]
        self.corpus_cache = [self.corpus_cache[i] for i in corpus_sorted_idx]

        query_sorted_idx = sorted(
            range(len(self.query_path)), key=lambda i: self.query_path[i]
        )
        self.query_path = [self.query_path[i] for i in query_sorted_idx]
        self.query_cache = [self.query_cache[i] for i in query_sorted_idx]

        self._download_hf_hub_uri()

    def _download_hf_hub_uri(self):
        """Replaces huggingface hub URIs with path to downloaded local files.

        This method expects a python list for the following attributes: ``query_path``,
        ``corpus_path``, ``qrel_path``, and ``query_subset_path``.
        """
        fs = None

        def download_hf_hub_uri(path: Union[str, Path]) -> Union[str, Path]:
            # Download an hf hub URI and return its path
            if not isinstance(path, str) or not path.startswith("hf://"):
                return path

            nonlocal fs
            if fs is None:
                fs = hf_hub.HfFileSystem()

            info = fs.resolve_path(path)
            if info.path_in_repo == "":
                msg = f"The path must point to a file in the remote repo. Got: '{info.path_in_repo}'"
                raise ValueError(msg)

            local_path = hf_hub.hf_hub_download(
                repo_id=info.repo_id,
                repo_type=info.repo_type,
                revision=info.revision,
                filename=info.path_in_repo,
            )
            return local_path

        for path_list in [
            self.qrel_path,
            self.query_path,
            self.corpus_path,
            self.query_subset_path,
        ]:
            if path_list is None:
                continue
            for i in range(len(path_list)):
                path_list[i] = download_hf_hub_uri(path_list[i])

    def to_dict(self) -> Dict:
        """Return a json serializable view of the class attributes."""
        self.ensure_list_of_correct_dtype()
        json_dict = dict()
        field_names = [f.name for f in fields(self)]
        for fname in field_names:
            fvalue = getattr(self, fname)
            if fname in [
                "qrel_path",
                "corpus_path",
                "query_path",
                "corpus_cache",
                "query_cache",
                "query_subset_path",
            ]:
                if fvalue is not None:
                    # make sure paths are of type 'str'
                    fvalue = [v if v is None else str(v) for v in fvalue]
            if fname in ["filter_fn", "score_transform", "group_filter_fn"]:
                if fvalue is not None and callable(fvalue):
                    # for functions, just get their str representations
                    fvalue = str(fvalue)
            json_dict[fname] = fvalue
        return json_dict
