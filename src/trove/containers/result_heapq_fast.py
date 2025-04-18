import copy
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch

from ..logging_utils import get_logger_with_config

logger, logging_conf = get_logger_with_config("trove")


class FastResultHeapq:
    def __init__(
        self,
        topk: int = 100,
        special_docids: Optional[
            Dict[str, Union[List[str], Dict[str, Union[int, float]]]]
        ] = None,
    ) -> None:
        """keeps track of the topk largest scores for each query.

        This is much faster than the simple python heapq used in :class:`~trove.containers.result_heapq.ResultHeapq`.
        This class requires to iterate over document and query batches in nested for loops,
        and crucially, the **outer loop generates the document batches**. I.e.,

        .. code-block:: python

            for doc_batch in corpus:
                for query_batch in queries:
                    fast_result_heapq(scores, ...)

        It keeps track of the topk most similar documents seen so far. It buffers
        all score batches for each ``doc_batch`` and merges them with results collected
        so far right before moving on to the next ``doc_batch``.

        Unlike ``ResultHeapq``, it uses GPUs for computation if it receives tensors already on GPU.


        This class does not provide all the nice utilities for input/output formatting that ``ResultHeapq`` does.
        To use those utilities, after you are done collecting the similarities, export the data from this class
        and import it into a fresh instance of ``ResultHeapq``.

        The arguments are the same as :class:`~trove.containers.result_heapq.ResultHeapq`. See :class:`~trove.containers.result_heapq.ResultHeapq` docstring for details.
        """
        self.topk = topk

        # Holds topk docs seen so far (except current batch)
        # each element of topk_docs_so_far['indices'] corresponds to one `query_batch`
        # Same for topk_docs_so_far['scores']
        self.topk_docs_so_far = None
        # Same structure as `self.topk_docs_so_far`
        # Collects topk documents from current `doc_batch`
        self.topk_docs_batch = None
        # all docids in the order that they are iterated over
        # It is a flat list of docids.
        self.all_docids = None
        # It is a nested list, where each element is the list of qids for the corresponding `query_batch`
        self.all_qids = None
        # docids for the current 'doc_batch'
        self.current_doc_batch_ids = None
        # mapping from document id to its index in the current 'doc_batch'
        self.current_doc_batch_id2idx = None
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

    def export_result_dump(
        self, reset_state: bool = False
    ) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
        """Export all the results collected so far with the same format as ``ResultHeapq``

        see :class:`~trove.containers.result_heapq.ResultHeapq` for details.
        """
        # Make sure all the collected similarity scores are merged into the final topk documents.
        self.update_topk_records()

        topk_docs = defaultdict(list)
        for q_batch_idx in range(len(self.topk_docs_so_far["indices"])):
            qids = self.all_qids[q_batch_idx]
            topk_scores = self.topk_docs_so_far["scores"][q_batch_idx]
            topk_scores = topk_scores.detach().cpu().tolist()
            topk_indices = self.topk_docs_so_far["indices"][q_batch_idx]
            topk_indices = topk_indices.detach().cpu().tolist()
            topk_docids = [
                [self.all_docids[idx] for idx in row_indices]
                for row_indices in topk_indices
            ]

            for qid, scores, docids in zip(qids, topk_scores, topk_docids):
                topk_docs[qid].extend(zip(scores, docids))
        special_docs = copy.deepcopy(self.special_docs)
        results = {"topk_docs": dict(topk_docs), "special_docs": special_docs}
        if reset_state:
            self.reset_state()
        return results

    def reset_state(self) -> None:
        """Clear the data collected so far."""
        self.topk_docs_so_far = dict(indices=[], scores=[])
        self.topk_docs_batch = dict(indices=[], scores=[])
        self.all_docids = list()
        self.all_qids = list()
        self.special_docs = dict(qids=[], docids=[], scores=[])
        self.current_doc_batch_ids = None
        self.current_doc_batch_id2idx = None

    def update_topk_records(self) -> None:
        """Merges topk records collected so far with topk records from the most recent batch
        selects the new topk records."""
        if not len(self.topk_docs_batch["indices"]):
            # It is the very first query_batch. We have not collected anything yet.
            return

        if not len(self.topk_docs_so_far["indices"]):
            # These are scores for the very doc_batch. There is nothing to merge them with.
            self.topk_docs_so_far["indices"] = self.topk_docs_batch["indices"]
            self.topk_docs_so_far["scores"] = self.topk_docs_batch["scores"]
            self.topk_docs_batch = dict(indices=[], scores=[])
            return

        # Sanity check. They should all have the same number of query batches.
        assert all(
            (
                len(self.topk_docs_so_far["scores"])
                == len(self.topk_docs_so_far["indices"]),
                len(self.topk_docs_batch["indices"])
                == len(self.topk_docs_so_far["indices"]),
                len(self.topk_docs_batch["scores"])
                == len(self.topk_docs_so_far["indices"]),
            )
        )

        for q_batch_idx in range(len(self.topk_docs_so_far["indices"])):
            prev_scores = self.topk_docs_so_far["scores"][q_batch_idx]
            prev_indices = self.topk_docs_so_far["indices"][q_batch_idx]

            batch_scores = self.topk_docs_batch["scores"][q_batch_idx]
            batch_indices = self.topk_docs_batch["indices"][q_batch_idx]

            # update batch index to account for documents seen in previous batches.
            batch_indices = (
                batch_indices + len(self.all_docids) - len(self.current_doc_batch_ids)
            )

            all_scores = torch.concat([prev_scores, batch_scores], dim=1)
            all_indices = torch.concat([prev_indices, batch_indices], dim=1)

            new_topk_scores, tmp_topk_indices = torch.topk(
                all_scores,
                k=min(self.topk, all_scores.shape[1]),
                dim=-1,
                largest=True,
                sorted=True,
            )
            new_topk_indices = all_indices[
                torch.arange(
                    all_indices.shape[0], device=all_indices.device, dtype=torch.long
                )[:, None],
                tmp_topk_indices,
            ]

            self.topk_docs_so_far["scores"][q_batch_idx] = new_topk_scores
            self.topk_docs_so_far["indices"][q_batch_idx] = new_topk_indices

        self.topk_docs_batch = dict(indices=[], scores=[])

    def __call__(
        self,
        scores: torch.Tensor,
        qids: List[str],
        docids: List[str],
        is_first_query_batch: bool,
        **kwargs,
    ) -> None:
        """Adds the calculated scores between a batch of queries and passages.

        Intended to be used as a callable in a pytorch evaluation loop.

        **remember to call :meth:`self.update_topk_records` for the very last manually.**
        It is not done automatically for the last batch.

        Args:
            scores (torch.Tensor): query document similarity scores. Shape: ``[NUM_QUERIES, NUM_DOCS]``
            qids (List[str]): List of query ids in the batch of length ``NUM_QUERIES``
            docids (List[str]): List of document ids in the batch of length ``NUM_DOCS``
            is_first_query_batch (bool): **Must** be set true, if we are processing the very first
                query batch for the current document batch..
            **kwargs: No effect. Just for compatibility with other classes.
        """
        if is_first_query_batch:
            # It is the first query_batch for a new doc_batch.
            # First, merge the results collected for previous 'doc_batch' before processing the new 'doc_batch'
            self.update_topk_records()
            if len(self.special_docids):
                self.current_doc_batch_id2idx = {_id: i for i, _id in enumerate(docids)}
            self.current_doc_batch_ids = docids
            self.all_docids.extend(docids)

        if not len(self.topk_docs_so_far["indices"]):
            # It is the very first 'doc_batch'. qids repeat in the same order for all doc batches.
            # Collect them only once.
            self.all_qids.append(qids)

        topk_scores, topk_indices = torch.topk(
            scores, k=min(self.topk, scores.shape[1]), dim=-1, largest=True, sorted=True
        )
        self.topk_docs_batch["scores"].append(topk_scores)
        self.topk_docs_batch["indices"].append(topk_indices)

        if len(self.special_docids):
            doc_id2idx = self.current_doc_batch_id2idx
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
