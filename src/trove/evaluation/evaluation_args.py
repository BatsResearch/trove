from dataclasses import dataclass
from typing import Optional

import torch.distributed as dist
from transformers import TrainingArguments

from ..logging_utils import get_logger_with_config


@dataclass
class EvaluationArguments(TrainingArguments):
    output_dir: Optional[str] = None
    """Directory to save the results."""
    per_device_matmul_batch_size: int = 256
    """Batch size for score calculation operation (i.e., ``matmul(q, doc)``).
    We use ``per_device_eval_batch_size`` as batch size for encoding the query and documents.
    """
    precompute_corpus_embs: Optional[bool] = False
    """Precompute the corpus embeddings, write to cache, and then calculate their score with query embeddings.
    They will be written to cache anyways (either temporarily or permanently).
    This option just controls if we should finish encoding documents before calculating the scores.
    """
    encoding_cache_dir: Optional[str] = None
    """If provided, write the embedding vectors to this directory.
    This option will be ignored for any ``EncodingDataset`` that has a cache filename attached to it already.
    If not provided, cache is written to a temporary directory and deleted before exit.
    """
    ir_metrics_k_values: Optional[str] = "1,3,5,10,100"
    """A comma separated list of cutoff values for IR metrics.
    **NOTE**: It is only used if ``compute_metrics`` is not passed to ``RetrievalEvaluator.__init__()`` method.
    """
    ir_metrics_relevance_threshold: Optional[int] = None
    """Minimum groundtruth relevancy level (inclusive) for a document to be considered relevant when calculating IR metrics.
    If not ``None``, it is passed to :class:`~trove.evaluation.metrics.IRMetrics` init method. See its docstring for details.
    **NOTE**: It is only used if ``compute_metrics`` is not passed to ``RetrievalEvaluator.__init__`` method.
    """
    search_topk: Optional[int] = None
    """Number of documents to retrieve during nearest neighbor search. Must be ``>= max(ir_metrics_k_values)``.
    Defaults to ``max(ir_metrics_k_values)``. This is useful to select the number of mined hard negatives for each query."""
    no_annot_in_mined_hn: bool = True
    """If true, annotated documents are **not** included in hard negative mining results, even if they are annotated as irrelevant.
    I.e., documents with groundtruth relevance label of zero are also excluded from hard negative mining results. If false, all documents are used for hard negative mining."""
    merge_mined_qrels: bool = False
    """By default, hard negative mining results are saved in a separate file for each pair of input query-corpus files.
    For example if you have two query files and two corpus files, you will end up with four qrel files.
    This allows you to combine multiple query (corpus) files that use the same ID different queries (documents).
    Set this to ``True`` to write all hard negative mining results in one file. It will raise an exception if it finds two records with the same ID.
    """
    pbar_mode: Optional[str] = "all"
    """Determines which processes show the progress bar. You can choose from one of ``['none', 'main', 'local_main', and 'all']`` values."""
    print_mode: Optional[str] = "main"
    """Determines which processes can print to stdout. You can choose from one of ``['none', 'main', 'local_main', and 'all']`` values."""
    cleanup_temp_artifacts: bool = True
    """If true, it removes all embedding cache files that it has generated but the user did not ask to save them explicitly."""
    save_eval_topk_logits: bool = False
    """If true, save the score of topk retrieved docs for each query during evaluation to disk. Note that if you have multiple query (or document) files that share the same ID, setting this to ``True`` will raise an exception."""
    output_qrel_format: str = "tsv"
    """The format and structure of the output file that search results (qid, docid, and scores) will be written to.
    There are two options:

        * ``tsv`` is the standard qrel format in a tsv file with three columns: `query-id`, `corpus-id`, and `score`
        * ``grouped`` is a json lines file where each record has three keys. ``qid`` is the ID of the current query.
          ``docid`` and ``score`` are two lists of the same size that hold the ID and score of the related documents for ``qid``
    """
    fair_sharding: bool = False
    """(Only used in distributed environments) If false, shard the dataset into chunks of roughly equal sizes.
    If true, shard the dataset such that devices with higher throughput are assigned bigger shards.
    This is to avoid idle GPU cycles when mixing GPUs with different capabilities.
    """
    broadcast_output: bool = True
    """(only for distributed environments) If true, the output of ``RetrievalEvaluator.evaluate()``
    and ``RetrievalEvaluator.mine_hard_negatives()`` are duplicated across all processes
    (i.e., these methods return identical outputs in all processes). If false, only
    the main process returns the output and other processes return ``None``. Set it to ``False`` to
    save memory on machines with multiple GPUs.
    """
    trove_logging_mode: str = "all"
    """Determines which processes can use the logging module. It is just a soft limit:
    the excluded processes can still log messages but their logging level is set to ERROR.
    You can choose from one of ``['main', 'local_main', and 'all']`` values.
    """

    def __post_init__(self):
        super().__post_init__()

        if self.trove_logging_mode not in ["main", "local_main", "all"]:
            msg = "Valid values for mode configs (print and pbar) are ['main', 'local_main', and 'all']"
            raise ValueError(msg)

        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        if world_size < 2:
            return
        log_level = None
        if (self.trove_logging_mode == "main" and self.process_index != 0) or (
            self.trove_logging_mode == "local_main" and self.local_process_index != 0
        ):
            log_level = "ERROR"
        get_logger_with_config(
            name="trove", log_level=log_level, rank=self.process_index, force=True
        )
