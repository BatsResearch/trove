import copy
import heapq
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch

from ..logging_utils import get_logger_with_config

logger, logging_conf = get_logger_with_config("trove")


class ResultHeapq:
    def __init__(
        self,
        topk: int = 100,
        special_docids: Optional[
            Dict[str, Union[List[str], Dict[str, Union[int, float]]]]
        ] = None,
    ) -> None:
        """A simple class that uses heapq to keep track of the topk largest scores for each query.

        You can also use an instance of this class as a callable in a pytorch evaluation loop.

        Args:
            topk (int): Number of items to hold in the heapq for each query.
            special_docids (Optional[Dict]): special set of documents for each query
                to collect their similarity scores regardless of their ranking.
                Do not use unless necessary. It degrades the performance.
                It can be in qrel format (i.e., ``special_docids[qid][docid]=some_score``). score is ignored here.
                It can also be a mapping from qid to a list of docids, i.e., ``special_docids[qid] = [docid1, docid2, ...]``
        """
        self.topk = topk
        self.topk_docs = None
        self.special_docs = None
        self.reset_state()

        self.special_docids = defaultdict(set)
        if special_docids is not None:
            for qid in special_docids.keys():
                _docids = special_docids[qid]
                if isinstance(_docids, dict):
                    # special_docids is in nested_qrel format
                    _docids = list(_docids.keys())
                self.special_docids[qid].update(_docids)
        self.special_docids = dict(self.special_docids)

    def reset_state(self) -> None:
        """Clear the data collected so far."""
        self.topk_docs = defaultdict(list)
        self.special_docs = dict(qids=[], docids=[], scores=[])

    def add_triplet(self, qid: str, docid: str, score: Union[int, float]) -> None:
        """Add (docid, score) to the heapq for qid.

        Args:
            qid (str): query id
            docid (str): document id
            score (Union[int, float]): similarity score between query and document.
        """
        if len(self.topk_docs[qid]) < self.topk:
            heapq.heappush(self.topk_docs[qid], (score, docid))
        else:
            heapq.heappushpop(self.topk_docs[qid], (score, docid))

    def add_qrel_nested_dict(
        self,
        qrel_dict: Union[
            Dict[str, Dict[str, Union[int, float]]],
            Dict[str, Dict[str, Dict[str, Union[int, float]]]],
        ],
    ) -> None:
        """Import results in nested qrel dict format.

        The input is similar to the output of :meth:`as_qrel_nested_dict` method.

        Args:
            qrel_dict (Dict): results in nested qrel dict format,
                where ``qrel_dict[qid][docid]`` is the similarity score between ``qid`` and ``docid``.
                If ``qrel_dict`` contains keys ``topk_docs`` or ``special_docs``, it assumes that each of the
                ``qrel_dict['topk_docs']`` and ``qrel_dict['special_docs']`` is a separate qrel object and is merged into the corresponding internal collection.
                If none of these keys is available, it assumes that ``qrel_dict`` is one qrel object and is merged into ``self.topk_docs``.
        """
        _topk_docs = qrel_dict.get("topk_docs", None)
        _special_docs = qrel_dict.get("special_docs", None)
        if _topk_docs is None and _special_docs is None:
            _topk_docs = qrel_dict

        if _topk_docs is not None:
            for qid, qdata in _topk_docs.items():
                for docid, docscore in qdata.items():
                    self.add_triplet(qid=qid, docid=docid, score=docscore)

        if _special_docs is not None:
            for qid, qdata in _special_docs.items():
                self.special_docs["qids"].extend([qid] * len(qdata))
                self.special_docs["docids"].extend(qdata.keys())
                self.special_docs["scores"].extend(qdata.values())

    def __call__(
        self, scores: torch.Tensor, qids: List[str], docids: List[str], **kwargs
    ) -> None:
        """Adds the calculated scores between a batch of queries and passages.

        Intended to be used as a callable in a pytorch evaluation loop.

        Args:
            scores (torch.Tensor): similarity scores between query and documents.
                2D tensor of shape ``[NUM_QUERIES, NUM_DOCS]``
            qids (List[str]): List of query ids in the batch of length NUM_QUERIES
            docids (List[str]): List of document ids in the batch of length NUM_DOCS
            **kwargs: No effect. Just for compatibility with other classes.
        """
        # Iterating over all scores takes a long time
        # Instead use torch to extract the topk largest scores and only add those to the heapq
        topk_scores, topk_indices = torch.topk(
            scores, k=min(self.topk, scores.shape[1]), dim=-1, largest=True, sorted=True
        )
        topk_scores = topk_scores.cpu().tolist()
        topk_indices = topk_indices.cpu().tolist()

        for q_id, q_scores, q_doc_indices in zip(qids, topk_scores, topk_indices):
            for score, d_idx in zip(q_scores, q_doc_indices):
                self.add_triplet(qid=q_id, docid=docids[d_idx], score=score)

        if len(self.special_docids):
            doc_id2idx = {_id: i for i, _id in enumerate(docids)}
            row_indices = list()
            col_indices = list()
            for qidx, qid in enumerate(qids):
                if qid not in self.special_docids:
                    continue
                _cell_docids = list(self.special_docids[qid].intersection(doc_id2idx))
                if not _cell_docids:
                    continue
                _col_indices = [doc_id2idx[_id] for _id in _cell_docids]
                col_indices.extend(_col_indices)
                row_indices.extend([qidx] * len(_cell_docids))
                self.special_docs["qids"].extend([qid] * len(_cell_docids))
                self.special_docs["docids"].extend(_cell_docids)

            if len(row_indices):
                cell_scores = scores[row_indices, col_indices].cpu().tolist()
                self.special_docs["scores"].extend(cell_scores)

    def as_qrel_nested_dict(
        self, collection: Optional[str] = None
    ) -> Union[Dict[str, Dict[str, float]], Dict[str, Dict[str, Dict[str, float]]]]:
        """Export the collected results as nested qrel dicts.

        Args:
            collection (Optional[Union[str, List[str]]]): The type of results to return.
                It can be one of the following:

                * ``None`` : Just return topk most similar documents
                * ``'topk_docs'`` : similar to None
                * ``'special_docs'`` : just return the collected results for special documents
                * ``'all'`` : return a dict with keys ``'special_docs'`` and ``'topk_docs'`` containing the corresponding results

        Returns:
            The collected similarities so far in qrel format (i.e., ``qrel[qid][docid]=sim(qid, docid)``).
            If asked for multiple collections of results, it returns a mapping from collection type to qrel.
        """
        logger.debug("Export collected scores in qrel nested dict format.")

        res = dict()

        if collection is None or collection in ["topk_docs", "all"]:
            qrel_mapping = defaultdict(dict)
            for qid, qdata in self.topk_docs.items():
                for docscore, docid in qdata:
                    qrel_mapping[qid][docid] = docscore
            res["topk_docs"] = dict(qrel_mapping)

        if collection in ["special_docs", "all"]:
            qrel_mapping = defaultdict(dict)
            for qid, docid, docscore in zip(
                self.special_docs["qids"],
                self.special_docs["docids"],
                self.special_docs["scores"],
            ):
                qrel_mapping[qid][docid] = docscore
            res["special_docs"] = dict(qrel_mapping)

        if len(res) == 1:
            return list(res.values())[0]
        else:
            return res

    def as_sorted_lists(self, reverse: bool = False) -> Dict[str, List[Tuple]]:
        """Export the topk collected so far as a sorted list for each query.

        Args:
            reverse (bool): Is used to sort the scores like ``sorted(..., reversed=reversed)``

        Returns:
            a mapping from ``qid`` to a list of tuples of ``(doc_score, doc_id)``. Each list is sorted based on the ``doc_score`` field.
        """
        output = dict()
        for qid, qdata in self.topk_docs.items():
            output[qid] = sorted(qdata, key=lambda x: x[0], reverse=reverse)
        return output

    def export_result_dump(
        self, reset_state: bool = False
    ) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
        """Export all the results collected so far in the same format as stored in this class.

        You can use this to export and merge the collected results from multiple collections,
        for example if you are doing a sharded evaluation in a distributed environment.

        Args:
            reset_state (bool): If true, clear the data collected so far.

        Returns:
            A dict with keys ``'topk_docs'`` and ``'special_docs'`` which hold the content of attributes of this class with the same name.
        """
        results = dict()
        # deepcopy the results so the changes made by user do not impact the state of this class
        results["topk_docs"] = copy.deepcopy(self.topk_docs)
        results["special_docs"] = copy.deepcopy(self.special_docs)
        if reset_state:
            self.reset_state()
        return results

    def merge_result_dump(self, results: Dict) -> None:
        """Merge the results exported from another instance of this class with their
        :meth:`export_result_dump` method.

        Args:
            results (Dict): The exported results from the other instances of ``ResultHeapq``.
        """
        logger.debug("Merge dump of collected score from other ResultHeapq instances.")

        for qid, scores_ids in results["topk_docs"].items():
            for score, docid in scores_ids:
                self.add_triplet(qid=qid, docid=docid, score=score)

        if "special_docs" in results:
            self.special_docs["qids"].extend(results["special_docs"]["qids"])
            self.special_docs["docids"].extend(results["special_docs"]["docids"])
            self.special_docs["scores"].extend(results["special_docs"]["scores"])

    def get_state_dict(self) -> Dict:
        """Export all the attributes of this class in a dictionary.

        You can use the output of this method to recreate an identical instance of this class later.

        Returns:
            mapping from the name of attributes of this class to their value.
        """
        state_dict = dict()
        state_dict["topk"] = self.topk
        # deepcopy so user changes outside of this class won't corrupt the internal state
        state_dict["special_docids"] = copy.deepcopy(self.special_docids)
        state_dict["special_docs"] = copy.deepcopy(self.special_docs)
        state_dict["topk_docs"] = copy.deepcopy(self.topk_docs)
        return state_dict
