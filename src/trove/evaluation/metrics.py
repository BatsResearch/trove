"""Provides Utilities to calculate IR metrics given query-passage qrels and scoress."""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from transformers import EvalPrediction, TrainingArguments

from .relevance_evaluator import RelevanceEvaluatorPlus


class IRMetrics:
    def __init__(
        self,
        k_values: Optional[Union[int, List[int]]] = None,
        measures: Optional[Set[str]] = None,
        relevance_threshold: int = 1,
        include_in_batch_negs: bool = False,
    ) -> None:
        """Calculates IR metrics given qrels and scores.

        * you can either use this class by calling :meth:`compute` which just returns the average metrics and does not change the instance internal state.
        * You can call :meth:`add_batch` multiple times and then call :meth:`aggregate_results` to get the average metrics across all examples seen so far.
        * You can also pass an instance of this class to ``transformers.Trainer`` as the ``compute_metrics`` argument.
          But, you have to set ``batch_eval_metrics=True`` to be compatible with this class.

        This is how to use it::

            metric_calculator = IRMetrics(k_values=[10, 100])
            args = TrainingArguments(..., batch_eval_metrics=True, label_names=["YOUR_LABEL_NAME"], ...)
            trainer = Trainer(args=args, ..., compute_metrics=compute_metrics, ...)
            ...

        Args:
            k_values (Union[int, List[int]]): a single or a list of cutoff values to calculate IR metrics for.
                E.g., nDCG@K or recall@K. If provided, use a predefined set of IR metrics with these cutoff values.
            measures (Optional[Set[str]]): a set of measures used with ``pytrec_eval``. You should either specify ``measures`` directly or specify ``k_values``
                and we instantiate a default set of IR metrics with the given ``k_values``.
            relevance_threshold (int): minimum groundtruth relevancy level (inclusive) for a document to be considered relevant.
                Basically documents with relevancy levels smaller than this value will be excluded from the groundtruth.
                See see ``pytrec_eval.RelevanceEvaluator.__init__()`` for details.
            include_in_batch_negs (bool): If true, include the in-batch negatives in metric calculation if available.
                It is only used if an instance of the class is passed to ``transformers.Trainer`` for `compute_metrics` argument.
        """
        self.include_in_batch_negs = include_in_batch_negs
        if self.include_in_batch_negs:
            raise NotImplementedError

        if (k_values is None) == (measures is None):
            msg = f"You should specify one and only one of the 'k_values' or 'measures' arguments. Got measures: '{measures}' and k_values: '{k_values}'"
            raise ValueError(msg)

        self.relevance_threshold = relevance_threshold
        if measures is not None:
            self.metric_ids = measures
        else:
            if isinstance(k_values, int):
                k_values = [k_values]
            if not isinstance(k_values, list):
                raise TypeError
            # Identifier for IR Metrics to calculate. These are later used with pytrec_eval library
            map_string = "map_cut." + ",".join([str(k) for k in k_values])
            ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
            recall_string = "recall." + ",".join([str(k) for k in k_values])
            precision_string = "P." + ",".join([str(k) for k in k_values])
            mrr_string = "mrr_cut." + ",".join([str(k) for k in k_values])
            self.metric_ids = {
                map_string,
                ndcg_string,
                recall_string,
                precision_string,
                mrr_string,
            }

        # keep track of the sum of metrics for all examples seen so far.
        self.metric_sum = defaultdict(float)
        # number of examples seen so far
        self.num_examples = 0

    def reset_state(self) -> None:
        """Clear the metrics collected so far.

        Useful if you are using the same instance for multiple eval loops.
        """
        del self.metric_sum
        self.metric_sum = defaultdict(float)
        self.num_examples = 0

    def _compute_metrics_sum(
        self,
        scores: Dict[str, Dict[str, Union[int, float]]],
        qrels: Dict[str, Dict[str, int]],
    ) -> Tuple[Dict[str, float], int]:
        """Calculates IR metrics for the given qrels and scores and returns the sum of the metrics
        across all queries.

        This function does not modify the internal state of the class.

        Args:
            scores (Dict[str, Dict[str, Union[int, float]]]): a mapping where ``scores[qid][docid]`` is the predicted similarity score between query ``qid`` and document ``docid``
            qrels (Dict[str, Dict[str, int]]): The ground truth relevance scores. ``qrels[qid][docid]`` is the groundtruth similarity score for query ``qid`` and document ``docid``

        Returns:
            A tuple. First item is the sum of calculated IR metrics for the given examples.
            The second item is the number of examples passed to this function in the current call.
        """
        current_metric_sum = defaultdict(float)
        current_num_examples = 0

        evaluator = RelevanceEvaluatorPlus(
            qrels, self.metric_ids, self.relevance_threshold
        )
        all_evals = evaluator.evaluate(scores)

        # aggregate the metrics for queries in this function call
        for query_eval in all_evals.values():
            for metric_name, metric_value in query_eval.items():
                current_metric_sum[metric_name] += metric_value
            current_num_examples += 1

        return current_metric_sum, current_num_examples

    def compute(
        self,
        scores: Dict[str, Dict[str, Union[int, float]]],
        qrels: Dict[str, Dict[str, int]],
    ) -> Dict[str, float]:
        """Calculates the average IR metrics for the given scores and qrels. Does not modify the
        internal state of the class.

        Args:
            scores (Dict[str, Dict[str, Union[int, float]]]): a mapping where ``scores[qid][docid]`` is the predicted similarity score between query ``qid`` and document ``docid``
            qrels (Dict[str, Dict[str, int]]): The ground truth relevance scores. ``qrels[qid][docid]`` is the groundtruth similarity score for query ``qid`` and document ``docid``

        Returns:
            A mapping from metric name to the average value of the metric for the given examples.
        """
        metric_sums, num_samples = self._compute_metrics_sum(scores=scores, qrels=qrels)

        current_avg = dict()
        for k, v in metric_sums.items():
            current_avg[k] = v / num_samples
        return current_avg

    def add_batch(
        self,
        scores: Dict[str, Dict[str, Union[int, float]]],
        qrels: Dict[str, Dict[str, int]],
    ) -> Dict[str, float]:
        """Calculates the average metrics for the given scores and qrels.

        It also updates the sum of the metric values and total number of examples in the instance's internal state.
        However, in each individual call, it only returns the average metrics for examples in the given call and not across all examples seen so far.

        See :meth:`compute` for description of arguments.

        Returns:
            The average IR metrics for examples in this batch.
        """
        _sum, num_samples = self._compute_metrics_sum(scores=scores, qrels=qrels)
        # Calculate the current average metrics and add them to the internal state of the metrics as well
        current_avg = dict()
        self.num_examples += num_samples
        for k, v in _sum.items():
            self.metric_sum[k] += v
            current_avg[k] = v / num_samples

        return current_avg

    def aggregate_results(self) -> Dict[str, float]:
        """Calculates the average IR metrics across all examples seen so far."""
        avg = dict()
        for _name, _sum in self.metric_sum.items():
            avg[_name] = _sum / self.num_examples
        return avg

    def __call__(
        self, eval_pred: EvalPrediction, compute_result: Optional[bool] = None
    ) -> Dict[str, float]:
        """Calculates the IR metrics given the ``eval_pred`` results from ``transformers.Trainer``.

        This is intended to be used by ``transformers.Trainer`` to calculate the metrics for each batch in evaluation loop.

        * The input logits (i.e., similarity scores) are expected to be of shape `[NUM_QUERIES, NUM_QUERIES * NUM_DOCS_PER_QUERY]`
        * The columns are ordered according to queries. I.e., all passages related to first query show up first, then passages related to second query, and so on.
        * In a distributed environment where ``query_embs`` and ``passage_embs`` are gathered from all processes before calculating the similarities,
          the dimensions of the logits matrix are multiplied by world-size along each axis (Trove models do not gather scores from other processes during eval).
        * In a distributed setup, the model vertically stacks all the gathered queries from all processes before calculating the scores. Then, Trainer
          also stacks the computed scores vertically once returned by the model. So, we need to make sure to account for these duplicate rows in a distributed environment.
          (we don't need to worry about this since Trove models do not do this if model is in eval mode (i.e., ``assert not model.training``)).

        one more thing to be aware of: on each process, Trainer takes the logits from the model output and labels from the input (i.e., without model touching them).
        And then processes them as needed (e.g., gather and concat from other processes, etc.).
        Just be aware that if you modify the labels in your model, that change is not reflected in the labels that this function receives.

        Args:
            eval_pred (EvalPrediction): evaluation results passed by transformers.Trainer
            compute_result (Optional[bool]): If true, return the average metrics over all queries seen so far instead of metrics for the current batch.
                And also, reset the state. I.e., clear the metrics gathered so far.

        Returns:
            The calculated IR metrics. A mapping from metric_name to metric_value
        """
        labels = eval_pred.label_ids
        logits = eval_pred.predictions

        # remove the duplicate rows in a distributed environment. See docstring for more info
        logits = logits[: len(labels)]

        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits)

        # Extract the similarity docs between the query and its corresponding documents
        # if d_i_j is the j_th document for i_th query, the docs are ordered like:
        # [d_0_0, d_0_1, d_0_2, d_1_0, d_1_1, d_1_2, ...]
        # to select the subset of scores between queries and their corresponding documents, we create the following index
        # [[0, 1, 2],
        #  [3, 4, 5],
        #  ...
        #  ]
        # using these indices with torch.gather along dim 1 gives us the predicted sim scores

        num_queries, docs_per_query = labels.shape
        gather_idx = torch.arange(
            num_queries * docs_per_query, device=logits.device, dtype=torch.long
        )
        gather_idx = gather_idx.reshape(num_queries, docs_per_query)
        gather_idx = torch.remainder(gather_idx, logits.shape[1])
        logits = torch.gather(logits, dim=1, index=gather_idx)

        # <<<<<<<<<<<< alternative to the above operation that is slightly faster (very small difference) but more difficult to understand >>>>>>>
        # num_queries, docs_per_query = labels.shape
        # queries_per_batch = logits.shape[1] // docs_per_query
        # all_sub_logits = list()
        # for b_qidx in range(0, num_queries, queries_per_batch):
        #     sub_logits = logits[b_qidx: b_qidx + queries_per_batch]
        #     sub_logits = sub_logits.unfold(1, docs_per_query, docs_per_query).diagonal().T
        #     all_sub_logits.append(sub_logits)
        # logits = torch.vstack(all_sub_logits)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        logits = logits.detach().cpu().tolist()
        labels = labels.detach().cpu().tolist()

        scores = defaultdict(dict)
        qrels = defaultdict(dict)

        for q_idx, (gt_labels, pred_scores) in enumerate(zip(labels, logits)):
            for doc_idx, (gt_label, pred_score) in enumerate(
                zip(gt_labels, pred_scores)
            ):
                scores[str(q_idx)][str(doc_idx)] = pred_score
                qrels[str(q_idx)][str(doc_idx)] = gt_label

        scores = dict(scores)
        qrels = dict(qrels)
        results = self.add_batch(scores=scores, qrels=qrels)

        if compute_result:
            # Average across all examples seen so far
            results = self.aggregate_results()
            self.reset_state()

        return results

    def check_training_arguments(self, args: TrainingArguments):
        """If used with ``transformers.Trainer``, check that we support the arguments that it
        uses."""
        if not args.batch_eval_metrics:
            raise NotImplementedError(
                "Calculating metrics for the aggregated prediction results is not supported."
            )
