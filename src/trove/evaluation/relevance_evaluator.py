from collections import defaultdict
from typing import Dict, List, Set, Union

import pytrec_eval

from .mrr import MRRCutEvaluator


class RelevanceEvaluatorPlus:
    def __init__(
        self,
        query_relevance: Dict[str, Dict[str, Dict[str, int]]],
        measures: Union[Set[str], List[str]],
        relevance_level: int = 1,
    ) -> None:
        """Expands ``pytrec_eval.RelevanceEvaluator`` with new metrics.

        User facing API is the same as ``pytrec_eval.RelevanceEvaluator`` and should be a drop-in replacement.
        See ``pytrec_eval.RelevanceEvaluator`` for more details.


        Args:
            query_relevance (Dict[str, Dict[str, Dict[str, int]]]): see ``pytrec_eval.RelevanceEvaluator.__init__()``
            measures (Union[Set[str], List[str]]): see ``pytrec_eval.RelevanceEvaluator.__init__()``
            relevance_level (int): see ``pytrec_eval.RelevanceEvaluator.__init__()``
        """
        _new_evaluators = [MRRCutEvaluator]
        basename_to_evaluator = {_cls.measure_name: _cls for _cls in _new_evaluators}
        # Requested measures for new Evaluators
        basename_to_measures = {k: set() for k in basename_to_evaluator.keys()}

        pytrec_measures = set()
        # We assume none of the new Evaluators define the same base measure name as the ones supported by pytrec_eval.
        for measure in measures:
            for bname in basename_to_measures.keys():
                if measure.startswith(bname):
                    basename_to_measures[bname].add(measure)
                    break
            else:
                # We assume pytrec_eval supports the measure if it is not supported by any of the new Evaluators
                pytrec_measures.add(measure)

        if len(pytrec_measures) != 0:
            self.pytrec_evaluator = pytrec_eval.RelevanceEvaluator(
                query_relevance=query_relevance,
                measures=pytrec_measures,
                relevance_level=relevance_level,
            )
        else:
            self.pytrec_evaluator = None

        self.other_evaluators = list()
        for other_bname, other_measures in basename_to_measures.items():
            if len(other_measures) != 0:
                other_eval_cls = basename_to_evaluator[other_bname]
                self.other_evaluators.append(
                    other_eval_cls(
                        query_relevance=query_relevance,
                        measures=other_measures,
                        relevance_level=relevance_level,
                    )
                )

    def evaluate(
        self, scores: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Calculate IR metrics for the given similarity scores.

        User facing API is the same as ``pytrec_eval.RelevanceEvaluator.evaluate()`` and should be a drop-in replacement.
        See ``pytrec_eval.RelevanceEvaluator.evaluate()`` for more details.

        Args:
            scores (Dict[str, Dict[str, Dict[str, float]]]): predicted similarity scores.
                see ``pytrec_eval.RelevanceEvaluator.evaluate()``
        Returns:
            a mapping where ``output[qid][metric_name]`` is the value of "metric_name" for query "qid".
            see ``pytrec_eval.RelevanceEvaluator.evaluate()``
        """
        if self.pytrec_evaluator is not None:
            metrics = self.pytrec_evaluator.evaluate(scores)
        else:
            metrics = defaultdict()

        for other_eval in self.other_evaluators:
            other_metrics = other_eval.evaluate(scores)
            # merge with other metrics already calculated
            for qid, qmetrics in other_metrics.items():
                metrics[qid] = {**metrics[qid], **qmetrics}
        if isinstance(metrics, defaultdict):
            metrics = dict(metrics)
        return metrics
