from typing import Dict, List, Set, Union

from . import metric_utils


class MRRCutEvaluator:
    measure_name: str = "mrr_cut"

    def __init__(
        self,
        query_relevance: Dict[str, Dict[str, Dict[str, int]]],
        measures: Union[str, List[str], Set[str]],
        relevance_level: int = 1,
    ) -> None:
        """Calculate MRR with cutoff values.

        The basename for this metric is ``mrr_cut``. The user facing API mimics that of ``pytrec_eval.RelevanceEvaluator``.

        Args:
            query_relevance (Dict[str, Dict[str, Dict[str, int]]]): See ``pytrec_eval.RelevanceEvaluator.__init__``.
            measures (Union[str, List[str], Set[str]]): See ``pytrec_eval.RelevanceEvaluator.__init__``.
            relevance_level (int): docs with ground truth relevancy level ``>= relevance_level`` are considered relevant.
                See ``pytrec_eval.RelevanceEvaluator.__init__``.
        """
        measures = metric_utils.expand_and_combine_eval_measures(
            measures=measures,
            supported_measures=self.__class__.measure_name,
            reconstruct=False,
        )
        if len(measures[self.__class__.measure_name]) == 0:
            msg = "There are no default cutoff values for 'mrr_cut'. Please specify cutoff values like 'mrr_cut.k1,k2,k3'."
            raise ValueError(msg)
        k_values = list(measures[self.__class__.measure_name])
        for k in k_values:
            if int(k) != float(k):
                msg = f"cutoff values for 'mrr_cut' must be integers. Got '{k}'"
                raise ValueError(msg)
        self.k_values = [int(k) for k in k_values]

        # docs for each query that are considered related according to groundtruth qrels.
        self.related_docs = {
            qid: {_id for _id, _sc in qdata.items() if _sc >= relevance_level}
            for qid, qdata in query_relevance.items()
        }

    def evaluate(
        self, scores: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Calculate MRR for the given similarity scores.

        Same API as ``pytrec_eval.RelevanceEvaluator.evaluate()``

        Args:
            scores (Dict[str, Dict[str, Dict[str, float]]]): predicted similarity scores.
                Same as args for ``pytrec_eval.RelevanceEvaluator.evaluate()``.

        Returns:
            Calculated MRR values. ``ret[qid][mrr_cut_k]`` is the "MRR@k" for qid.
            The output structure is the same as ``pytrec_eval.RelevanceEvaluator.evaluate()``.
        """
        max_k = max(self.k_values)
        common_qids = set(self.related_docs).intersection(scores)

        query_mrrs = dict()  # holds MRR values for each query
        for qid in common_qids:
            qdata = scores[qid]
            sorted_docids = sorted(qdata, key=qdata.get, reverse=True)[:max_k]

            # 'rank' is the rank of the first related document for the query
            rank = max_k + 1
            for r, _id in enumerate(sorted_docids):
                if _id in self.related_docs[qid]:
                    rank = r + 1
                    # we are only interested in the rank of the first related document
                    break

            query_mrrs[qid] = dict()
            for k in self.k_values:
                m_name = f"{self.__class__.measure_name}_{k}"
                if k < rank:
                    query_mrrs[qid][m_name] = 0
                else:
                    query_mrrs[qid][m_name] = 1.0 / rank

        return query_mrrs
