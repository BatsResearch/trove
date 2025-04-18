import copy
import gc
import itertools
import json
import math
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase, default_data_collator

from .. import cache_manager, file_utils
from ..containers import FastResultHeapq, ResultHeapq
from ..data import EncodingDataset, MultiLevelDataset
from ..logging_utils import get_logger_with_config, rpath
from ..modeling import PretrainedRetriever, RetrieverOutput
from . import metric_utils
from .evaluation_args import EvaluationArguments
from .evaluator_mixin_utils import RetrievalEvaluatorUtilsMixin
from .metrics import IRMetrics

logger, logging_conf = get_logger_with_config("trove")


class RetrievalEvaluator(RetrievalEvaluatorUtilsMixin):
    def __init__(
        self,
        args: Optional[EvaluationArguments] = None,
        model: Union[PretrainedRetriever, PreTrainedModel, nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        data_collator: Optional[Callable] = None,
        eval_dataset: Optional[MultiLevelDataset] = None,
        compute_metrics: Optional[IRMetrics] = None,
        logit_collector: Optional[Union[ResultHeapq, FastResultHeapq]] = None,
        tracker_init_kwargs: Optional[Dict] = None,
        tracker_extra_configs: Optional[Union[List[Dict], Dict]] = None,
        tracker_callbacks: Optional[Union[Any, List[Any]]] = None,
    ) -> None:
        """Simple class for evaluating retrieval performance, mining hard negatives, or just
        computing the embeddings.

        Args:
            args (Optional[EvaluationArguments]): general arguments to control the evaluation/encoding process.
            model (Union[PretrainedRetriever, PreTrainedModel, nn.Module]): retriever model to use
            tokenizer (Optional[PreTrainedTokenizerBase]): (Not used currently)
            data_collator (Optional[Callable]): callable to create a batch from a list of examples.
                It should be able to tokenize the text if embeddings are not precomputed.
            eval_dataset (Optional[MultiLevelDataset]):
                Evaluate the performance on this dataset.
            compute_metrics (Optional[IRMetrics]):
                an instance of IRMetrics class to calculate the IR metrics from predicted scores and ground truth qrels.
            logit_collector (Optional[ResultHeapq]):
                an instance of ResultHeapq that should take the similarity scores for each batch
                and keep the topk most similar documents with their scores for each query.
            tracker_init_kwargs (Optional[Dict]): extra kwargs for initializing experiment trackers. See :class:`~trove.evaluation.evaluator_mixin_utils.RetrievalEvaluatorUtilsMixin` for details.
            tracker_extra_configs (Optional[Union[List[Dict], Dict]]): extra configs to log with experiment trackers. See :class:`~trove.evaluation.evaluator_mixin_utils.RetrievalEvaluatorUtilsMixin` for details.
            tracker_callbacks (Optional[Any]): One or multiple custom experiment tracker callbacks.
                See :class:`~trove.evaluation.evaluator_mixin_utils.RetrievalEvaluatorUtilsMixin` for details.
        """
        super().__init__(
            args=args,
            tracker_init_kwargs=tracker_init_kwargs,
            tracker_extra_configs=tracker_extra_configs,
            tracker_callbacks=tracker_callbacks,
        )

        self.model = model
        self.tokenizer = tokenizer

        if (
            compute_metrics is None
            and self.args.ir_metrics_k_values is not None
            and self.args.ir_metrics_k_values != ""
        ):
            _k_values = [
                int(ks.strip())
                for ks in self.args.ir_metrics_k_values.strip().split(",")
            ]
            _mkw = {}
            if self.args.ir_metrics_relevance_threshold is not None:
                _mkw["relevance_threshold"] = self.args.ir_metrics_relevance_threshold
            compute_metrics = IRMetrics(k_values=_k_values, **_mkw)
            del _mkw
            logger.debug(f"Create IRMetrics with these k_values: '{_k_values}'")
        self.compute_metrics = compute_metrics

        if data_collator is None:
            self.data_collator = default_data_collator
            logger.warning(
                "You are using `default_data_collator`. All your embeddings should be precomputed"
                " It will raise an exception if you try embed text."
            )
        else:
            self.data_collator = data_collator

        if logit_collector is None:
            _topk = None
            if self.args.search_topk is not None:
                _topk = self.args.search_topk
            elif self.compute_metrics is not None:
                _topk = metric_utils.get_measures_max_cutoff(
                    self.compute_metrics.metric_ids
                )
            if _topk is not None:
                logger.debug(f"Create FastResultHeapq with topk: {_topk}")
                logit_collector = FastResultHeapq(topk=_topk)
        self.logit_collector = logit_collector

        self.eval_dataset = eval_dataset

        if self.args.encoding_cache_dir is not None:
            self._shared_cache_uuid = None
            self.cache_pardir = self.args.encoding_cache_dir
        else:
            self._shared_cache_uuid = self.get_shared_uuid()
            self.cache_pardir = cache_manager.get_cache_pardir(
                artifact_content="embedding",
                artifact_type="final",
                cache_uuid=self._shared_cache_uuid,
            )
        logger.debug(
            f"Embeddings are cached in this directory if not specified otherwise: {rpath(self.cache_pardir)}"
        )

        # arguments used to instantiate torch DataLoader for encoding new documents
        self.dataloader_kwargs_encode = dict(
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            drop_last=False,
            shuffle=False,
            prefetch_factor=self.args.dataloader_prefetch_factor,
        )
        # arguments used to instantiate torch DataLoader reading embeddings from cache and calculating similarities
        self.dataloader_kwargs_matmul = {
            **self.dataloader_kwargs_encode,
            "batch_size": self.args.per_device_matmul_batch_size,
        }

        # In distributed environments, you get the option to shard the dataset based on device throughputs
        # and not evenly for all devices.
        self._shard_weights = None

        self.model = self.model.to(self.device)
        self.model.eval()

        self.initialize_trackers()

    def _forward_pass(self, *args, **kwargs) -> RetrieverOutput:
        """Call model's forward pass with necessary context managers."""
        with (
            torch.cuda.amp.autocast()
            if self.args.fp16 or self.args.bf16
            else nullcontext()
        ):
            with torch.no_grad():
                output = self.model(*args, **kwargs)
        return output

    def get_shard_weights(self, dataset: EncodingDataset) -> Optional[List[float]]:
        """Calculates relative shard sizes based on device performance.

        In each process, it runs the model on a small subset of the given dataset (identical across processes)
        and judges the model throughput based on the time that it takes to process this subset.
        Throughput in each process is used as the shard weight for that processes.

        Args:
            dataset (EncodingDataset): dataset to use for benchmarking. We only use a very small subset of it.

        Returns:
            ``None`` if fair sharding is disabled. A list of shard weights such that ``output[rank]`` is
            the shard weight for the process with ``rank``.
        """
        if not self.args.fair_sharding or not (
            self.is_distributed and self.world_size > 1
        ):
            # fair sharding disabled
            return None
        if self._shard_weights is not None:
            # We only compute them once in each session
            return self._shard_weights

        if self.all_devices_are_similar():
            # devices are the same. Practically, there is no fair sharding here
            # but we set identical shard weights that imply even sharding to avoid calling `self.all_devices_are_similar()` every time
            self._shard_weights = [1.0] * self.world_size
            logger.debug(f"shard weights: {self._shard_weights}")
            return self._shard_weights

        if isinstance(dataset, torch.utils.data.IterableDataset):
            msg = f"Fair sharding is not supported with Iterable datasets. Got: '${type(dataset)}"
            raise ValueError(msg)

        if (
            dataset.shard_idx is not None
            and dataset.shard_idx >= 0
            and dataset.num_shards is not None
            and dataset.num_shards > 0
        ):
            msg = (
                "'dataset' for computing fair shard weights must not be sharded to get"
                " consistent measurements across processes."
                " Got sharded dataset: '{dataset}'."
            )
            raise ValueError(msg)

        logger.info("Run benchmark to calculate shard weights.")

        # the type of data in 'EncodingDataset'
        if dataset.is_query:
            _key = "query"
        else:
            _key = "passage"

        # It is not only the GPU that impacts the performance. So, we choose num_samples for benchmarking
        # such that all dataloader workers have to run at least twice, which reduces the noise.
        num_samples = self.dataloader_kwargs_encode["batch_size"]
        if (
            self.dataloader_kwargs_encode["num_workers"]
            and self.dataloader_kwargs_encode["num_workers"] > 0
        ):
            # We want to have all workers occupied
            num_samples *= self.dataloader_kwargs_encode["num_workers"]
        if (
            self.dataloader_kwargs_encode["prefetch_factor"]
            and self.dataloader_kwargs_encode["prefetch_factor"] > 0
        ):
            # we want to prefetch as many as possible
            num_samples *= self.dataloader_kwargs_encode["prefetch_factor"]
        # double the number of samples to force them fetch new samples in the eval loop so we can measure the time
        num_samples = int(num_samples) * 2
        sample_indices = range(min(num_samples, len(dataset)))
        # repeat the sample indices again. If two processes are running on the same machine
        # It is possible that the first one reads them and they are held in CPU cache and
        # the second process gets a fake high throughput.
        # So we run through the samples twice in each process and just measure the time for the second pass.
        chosen_idx = list(
            itertools.islice(itertools.cycle(sample_indices), num_samples * 2)
        )

        logger.debug(f"Num samples in shard weight benchmark: {len(chosen_idx):,}")

        # we do not want cache
        with dataset.disable_cache():
            data_loader = DataLoader(
                dataset, sampler=chosen_idx, **self.dataloader_kwargs_encode
            )

            # index of the batch where the second pass starts and we start measuring time
            start_batch_idx = math.ceil(len(chosen_idx) / data_loader.batch_size) // 2
            start_time = None

            # It is nice to have them start at the same time. Not sure if it improves the benchmark or not.
            self.barrier()

            for i, batch in enumerate(
                self.pbar(data_loader, desc=f"Fair sharding measurements")
            ):
                if start_time is None and i >= start_batch_idx:
                    # start of the second pass
                    start_time = time.perf_counter()
                self._forward_pass(**{_key: batch[_key].to(self.device)})[_key]
            elapsed = time.perf_counter() - start_time
            # Just multiply it by 1000 so we do not end up with very small number in case the process is very slow
            throughput = 1000.0 / elapsed

        # throughput of all processes
        all_throughputs = self.all_gather_object(obj=throughput)
        self._shard_weights = all_throughputs
        logger.debug(f"shard weights: {self._shard_weights}")
        return self._shard_weights

    def _encode_one(self, encoding_ds: EncodingDataset, display_name: str) -> None:
        """Compute and cache the embeddings for a single ``EncodingDataset``.

        If in distributed environment, it shards the dataset across available processes.

        Args:
            encoding_ds (EncodingDataset): Compute and cache embeddings for this dataset.
            display_name (str): Name to use in console logs and progress bars.
        """
        _name = display_name  # short variable names are cleaner

        logger.info(f"Encoding {_name}")

        encoding_ds.load_cache()
        if encoding_ds.is_cache_available:
            # There is nothing new to do
            logger.debug("Using precomputed embeddings from cache.")
            return

        if not encoding_ds.is_writable:
            msg = (
                "It is not possible to write embeddings to cache,"
                " which means the calculated embeddings are thrown away without any use."
            )
            raise RuntimeError(msg)

        if self.is_distributed and self.world_size > 1:
            # (superficially) shard dataset in a distributed environment
            shard_weights = self.get_shard_weights(dataset=encoding_ds)
            encoding_ds.shard(
                shard_idx=self.rank,
                num_shards=self.world_size,
                shard_weights=shard_weights,
            )

        # the type of data in 'EncodingDataset'
        if encoding_ds.is_query:
            _key = "query"
        else:
            _key = "passage"

        data_loader = DataLoader(encoding_ds, **self.dataloader_kwargs_encode)

        # Make sure all processes know that they are creating a new cache file
        # If one creates a new file before others do the check, they assume corrupted cache and raise exception
        self.barrier()

        with encoding_ds.open_cache_io_streams():
            for batch in self.pbar(data_loader, desc=f"Encode {_name}"):
                emb = self._forward_pass(**{_key: batch[_key].to(self.device)})[_key]
                encoding_ds.cache_records(rec_id=batch["orig_rec_id"], value=emb)

        if self.is_distributed and self.world_size > 1:
            # Make sure all processes are done writing to cache
            self.barrier(infinite=True)  # barrier without timeout
            encoding_ds.unshard()

    def encode(
        self,
        eval_dataset: Optional[
            Union[
                MultiLevelDataset,
                Dict[str, MultiLevelDataset],
                List[MultiLevelDataset],
            ]
        ] = None,
        encoding_dataset: Optional[
            Union[EncodingDataset, Dict[str, EncodingDataset], List[EncodingDataset]]
        ] = None,
        cache_pardir: Optional[os.PathLike] = None,
        display_name: str = "eval",
    ) -> None:
        """Encode texts and cache embeddings.

        * For ``eval_dataset``, the query and corpus files that it uses are encoded.
        * If an ``EncodingDataset`` ends up without any cache filepath, it raises an
          exception (there is probably something wrong if you are just computing
          the embeddings and immediately throwing them away).

        Args:
            eval_dataset (Optional[Union[MultiLevelDataset, Dict, List]]): If given,
                encode the query and corpus files used in this dataset.
            encoding_dataset (Optional[Union[EncodingDataset, Dict, List]]): encode the data generated
                by these ``EncodingDataset`` instances.
            cache_pardir (Optional[os.PathLike]): Save the embedding cache here. The order of priority for
                where cache is saved is as following:

                    * cache file path already attached to EncodingDataset instances
                    * some file in ``cache_pardir`` given to this function
                    * some file in ``EvaluationArguments.encoding_cache_dir`` if provided
                    * no cache is saved (raises an exception)
            display_name (str): Name to use in console logs and progress bars.
                Ideally, it should contain some information about the dataset being encoded.
        """
        _name = display_name  # short variable names are cleaner

        if eval_dataset is not None and encoding_dataset is not None:
            msg = "You can pass either eval_dataset or encoding_dataset but not both."
            raise ValueError(msg)
        if eval_dataset is None and encoding_dataset is None:
            eval_dataset = self.eval_dataset
        if eval_dataset is None and encoding_dataset is None:
            msg = "There is no dataset to encode."
            raise ValueError(msg)

        if cache_pardir is None:
            # These encodings must be persistent,
            # So, use self.args.encoding_cache_pardir instead of self.cache_pardir
            cache_pardir = self.args.encoding_cache_dir

        if eval_dataset is not None:
            if isinstance(eval_dataset, dict):
                for _ds_name, _ds in eval_dataset.items():
                    _kw = dict(display_name=f"{_name}-{_ds_name}")
                    self.encode(eval_dataset=_ds, cache_pardir=cache_pardir, **_kw)
            elif isinstance(eval_dataset, list):
                for _ds_name, _ds in enumerate(eval_dataset):
                    _kw = dict(display_name=f"{_name}-{_ds_name}")
                    self.encode(eval_dataset=_ds, cache_pardir=cache_pardir, **_kw)
            else:
                query_ds, corpus_ds = eval_dataset.get_encoding_datasets(
                    encoding_cache_pardir=cache_pardir, load_cache_on_init=False
                )
                for _ds_idx, eds in enumerate(corpus_ds):
                    _kw = dict(display_name=f"{_name}-C{_ds_idx+1}-of-{len(corpus_ds)}")
                    self.encode(encoding_dataset=eds, cache_pardir=cache_pardir, **_kw)
                for _ds_idx, eds in enumerate(query_ds):
                    _kw = dict(display_name=f"{_name}-Q{_ds_idx+1}-of-{len(query_ds)}")
                    self.encode(encoding_dataset=eds, cache_pardir=cache_pardir, **_kw)
        else:
            if isinstance(encoding_dataset, dict):
                for _ds_name, _ds in encoding_dataset.items():
                    _kw = dict(display_name=f"{_name}-{_ds_name}")
                    self.encode(encoding_dataset=_ds, cache_pardir=cache_pardir, **_kw)
            elif isinstance(encoding_dataset, list):
                for _ds_name, _ds in enumerate(encoding_dataset):
                    _kw = dict(display_name=f"{_name}-{_ds_name}")
                    self.encode(encoding_dataset=_ds, cache_pardir=cache_pardir, **_kw)
            else:
                if encoding_dataset.cache_file_name is None:
                    # Create a unique cache file based on the data filename
                    assert cache_pardir is not None
                    _newfile = cache_manager.get_cache_dir(
                        input_data=encoding_dataset.filepath,
                        cache_pardir=cache_pardir,
                        cache_uuid=self._shared_cache_uuid,
                        write_metadata=self.rank == 0,
                    )
                    _newfile = _newfile.joinpath("embedding_cache.arrow").as_posix()
                    encoding_dataset.update_cache_file_name(_newfile, load=False)
                    logger.debug(f"New embedding cache file to set: {rpath(_newfile)}")
                self._encode_one(encoding_dataset, display_name=_name)
                # Free memory. We do not need the cache.
                encoding_dataset.unload_cache()

    def nearest_neighbor_search(
        self,
        query_dataset: Union[EncodingDataset, List[EncodingDataset]],
        corpus_dataset: Union[EncodingDataset, List[EncodingDataset]],
        logit_collector: Union[ResultHeapq, FastResultHeapq],
        cache_pardir: os.PathLike,
        display_name: str,
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """Find the nearest neighbors for each query.

        .. note::

            To save memory in distributed environments, this method only returns the output in process with ``rank == 0`` and returns ``None`` in other processes.

        Args:
            query_dataset (Union[EncodingDataset, List[EncodingDataset]]): One or multiple datasets holding
                the search queries.
            corpus_dataset (Union[EncodingDataset, List[EncodingDataset]]): One or multiple datasets holding
                the documents to search.
            logit_collector (Union[ResultHeapq, FastResultHeapq]): Instance of ``ResultHeapq`` or ``FastResultHeapq`` that takes the scores
                for each batch and keeps the topk most similar documents and their scores for each query.
            cache_pardir (os.PathLike): Write the embedding cache files to this directory.
            display_name (str): Name to use in console logs and progress bars.

        Returns:
            a mapping containing the collected topk similarities, which is the output of the ``logit_collector``.
        """
        _name = display_name  # short variable names are cleaner
        logger.info(f"Nearest neighbor search in: {_name}")

        assert logit_collector is not None
        # Make sure to start from a clean state and remove any scores from previous runs
        logit_collector.reset_state()

        query_datasets = (
            query_dataset if isinstance(query_dataset, List) else [query_dataset]
        )
        corpus_datasets = (
            corpus_dataset if isinstance(corpus_dataset, List) else [corpus_dataset]
        )

        for eds in query_datasets + corpus_datasets:
            if eds.cache_file_name is None:
                _newfile = cache_manager.get_cache_dir(
                    input_data=eds.filepath,
                    cache_pardir=cache_pardir,
                    cache_uuid=self._shared_cache_uuid,
                    write_metadata=self.rank == 0,
                )
                _newfile = _newfile.joinpath("embedding_cache.arrow").as_posix()
                eds.update_cache_file_name(file_name=_newfile, load=False)
                logger.debug(f"New embedding cache file to set: {rpath(_newfile)}")

        logger.info("Precompute query embeddings before calculating scores.")
        for i, eds in enumerate(query_datasets):
            self._encode_one(
                eds, display_name=f"{_name}-Q{i+1}-of-{len(query_datasets)}"
            )

        if self.args.precompute_corpus_embs:
            logger.info("Precompute corpus embeddings before calculating scores.")
            for i, eds in enumerate(corpus_datasets):
                self._encode_one(
                    eds, display_name=f"{_name}-C{i+1}-of-{len(corpus_datasets)}"
                )

        dataloader_kwargs_matmul = copy.deepcopy(self.dataloader_kwargs_matmul)
        dataloader_kwargs_encode = copy.deepcopy(self.dataloader_kwargs_encode)
        if self.args.precompute_corpus_embs:
            # If embeddings are precomputed, swap batch sizes of inner and outer loop
            # to speed up topk score calculations
            dataloader_kwargs_encode["batch_size"] = (
                self.args.per_device_matmul_batch_size
            )
            dataloader_kwargs_matmul["batch_size"] = (
                self.args.per_device_eval_batch_size
            )

        query_dataloaders = list()
        for qds in query_datasets:
            # Make sure cache is loaded. So we do not compute them twice or get errors
            qds.load_cache()
            # pre-fetch the query embeddings. It makes it much faster.
            query_dataloaders.append(
                [b for b in DataLoader(qds, **dataloader_kwargs_matmul)]
            )
            qds.unload_cache()  # free some memory
        gc.collect()  # force release memory

        # In the following we will loop over all query document pairs.
        # It seems complicated but it is basically doing this
        # for corpus_file in all_corpuses:
        #   for doc_batch in corpus_file:
        #       for query_file in all_query_files:
        #           for query_batch in query_file:
        #               calc sim(query_batch, doc_batch)

        for corpus_ds in self.pbar(
            corpus_datasets,
            desc=f"{_name} corpus files",
            disable=len(corpus_datasets) < 2,
        ):
            # Make sure cache is loaded. So we do not compute them twice or get errors
            corpus_ds.load_cache()
            if self.is_distributed and self.world_size > 1:
                # (superficially) shard dataset in a distributed environment
                shard_weights = self.get_shard_weights(dataset=corpus_ds)
                corpus_ds.shard(
                    shard_idx=self.rank,
                    num_shards=self.world_size,
                    shard_weights=shard_weights,
                )
            data_loader = DataLoader(corpus_ds, **dataloader_kwargs_encode)

            # Make sure all processes know that they are creating a new cache file
            # If one creates a new file before others do the check, they assume corrupted cache and raise exception
            self.barrier()
            with corpus_ds.open_cache_io_streams():
                for doc_batch in self.pbar(
                    data_loader, desc=f"{_name} docs", leave=len(corpus_datasets) < 2
                ):
                    if "embedding" in doc_batch:
                        doc_emb = doc_batch["embedding"].to(self.device)
                    else:
                        doc_emb = self._forward_pass(
                            **{"passage": doc_batch["passage"].to(self.device)}
                        )["passage"]
                        corpus_ds.cache_records(
                            rec_id=doc_batch["orig_rec_id"], value=doc_emb
                        )
                    # Marks the very first query batch that we process for each doc_batch
                    _is_first_query_batch = True
                    for q_loader in self.pbar(
                        query_dataloaders,
                        desc=f"{_name} query files",
                        disable=len(query_dataloaders) < 2,
                        leave=False,
                    ):
                        for q_batch in self.pbar(
                            q_loader,
                            desc=f"{_name} queries",
                            disable=len(q_loader) < 2,
                            leave=False,
                        ):
                            if "embedding" in q_batch:
                                q_emb = q_batch["embedding"].to(self.device)
                            else:
                                q_emb = self._forward_pass(
                                    **{"query": q_batch["query"].to(self.device)}
                                )["query"]

                            scores = self.model.similarity_fn(
                                query=q_emb, passage=doc_emb
                            )
                            logit_collector(
                                scores=scores,
                                qids=q_batch["rec_id"],
                                docids=doc_batch["rec_id"],
                                is_first_query_batch=_is_first_query_batch,
                            )
                            _is_first_query_batch = False

            if self.is_distributed and self.world_size > 1:
                # Make sure all processes are done writing to cache
                self.barrier()
                corpus_ds.unshard()
            # free up some memory
            corpus_ds.unload_cache()

        # free up some memory
        del doc_batch
        del doc_emb
        del q_loader
        del q_batch
        del q_emb
        del query_dataloaders
        gc.collect()

        if isinstance(logit_collector, FastResultHeapq):
            # Make sure to process the very last batch too.
            logit_collector.update_topk_records()
        logger.debug("Finished calculating scores.")

        if (self.is_distributed and self.world_size > 1) or not isinstance(
            logit_collector, ResultHeapq
        ):
            # Gather topk scores from all processes
            # Also make sure we are working with an instance of ResultHeapq since we later rely on import/export utilities that it provides
            logger.debug(
                "Export logit_collector dump to merge results from all processes and convert from 'FastResultHeapq' to 'ResultHeapq'"
            )
            collected_res = logit_collector.export_result_dump(reset_state=True)
            logger.info("Gathering scores from all processes.")
            print("Gathering scores from all processes.")
            all_collected_res = self.gather_object(obj=collected_res, dst=0)
            del collected_res  # save some memory
            gc.collect()

            if self.is_main_process:
                _logit_collector = ResultHeapq(
                    topk=logit_collector.topk,
                    special_docids=copy.deepcopy(logit_collector.special_docids),
                )
                del logit_collector  # save some memory

                while all_collected_res:
                    print("Merge scores from other devices.")
                    _logit_collector.merge_result_dump(all_collected_res.pop())

                collected_res = _logit_collector.as_qrel_nested_dict(collection="all")
                # save some memory
                _logit_collector.reset_state()
                del _logit_collector
            else:
                # save some memory
                logit_collector.reset_state()
                del logit_collector
                collected_res = None
                gc.collect()
        else:
            # There is only one process and we are working with an instance of ResultHeapq. so, no extra action needed
            collected_res = logit_collector.as_qrel_nested_dict(collection="all")
            logit_collector.reset_state()

        self.barrier(infinite=True)
        return collected_res

    def _evaluate_one(
        self,
        eval_dataset: MultiLevelDataset,
        logit_collector: Union[ResultHeapq, FastResultHeapq],
        cache_pardir: os.PathLike,
        display_name: str,
    ) -> Optional[Dict[str, Union[Any, Dict[str, float]]]]:
        """Run evaluation for a single dataset and return metrics and collected logits (i.e.,
        scores).

        .. note::

            To save memory in distributed environments, this method only returns the output in process with ``rank == 0`` and returns ``None`` in other processes.

        The intermediate embeddings are written to a temporary cache. If the user does not explicitly
        ask to cache the embeddings, the temporary cache is deleted before returning from this function.

        Args:
            eval_dataset (MultiLevelDataset): dataset to evaluate.
            logit_collector (Union[ResultHeapq, FastResultHeapq]): Instance of ``ResultHeapq`` or ``FastResultHeapq`` that takes the scores
                for each batch and keeps the topk most similar documents and their scores for each query.
            cache_pardir (os.PathLike): Write the embedding cache files to this directory.
            display_name (str): Name to use in console logs and progress bars.

        Returns:
            A mapping with two keys. ``metrics`` is the computed metrics; and
            ``logits`` is the subset of similarity scores collected by logit_collector.
        """
        logger.info(f"Evaluating: {display_name}")

        _metrics_max_cutoff = metric_utils.get_measures_max_cutoff(
            self.compute_metrics.metric_ids
        )
        if _metrics_max_cutoff > logit_collector.topk:
            msg = (
                "Number of retrieved document (i.e., 'logit_collector.topk') must be >= max cutoff value of metrics."
                f" Got number of retrieved documents: {logit_collector.topk} and metrics max cutoff value: {_metrics_max_cutoff}"
            )
            raise RuntimeError(msg)

        query_datasets, corpus_datasets = eval_dataset.get_encoding_datasets(
            encoding_cache_pardir=cache_pardir, load_cache_on_init=False
        )
        search_results = self.nearest_neighbor_search(
            query_dataset=query_datasets,
            corpus_dataset=corpus_datasets,
            logit_collector=logit_collector,
            cache_pardir=cache_pardir,
            display_name=display_name,
        )
        # free up some memory
        del query_datasets
        del corpus_datasets
        gc.collect()

        if not self.is_main_process:
            # to save memory, we only calculate the output and return it in the main process.
            # And avoid duplicate computation in other processes
            return None

        logger.debug("Prep Qrel and compute Metrics.")
        # Ground truth qrel
        gt_qrel = eval_dataset.get_qrel_nested_dict(return_global_ids=True)
        metrics = self.compute_metrics.compute(
            scores=search_results["topk_docs"], qrels=gt_qrel
        )

        output = {"metrics": metrics, "logits": search_results}
        return output

    @file_utils.call_cleanup("remove_temp_cache_pardir")
    def evaluate(
        self,
        eval_dataset: Optional[
            Union[MultiLevelDataset, Dict[str, MultiLevelDataset]]
        ] = None,
        logit_collector: Optional[
            Union[
                Union[ResultHeapq, FastResultHeapq],
                Dict[str, Union[ResultHeapq, FastResultHeapq]],
            ]
        ] = None,
        cache_pardir: Optional[os.PathLike] = None,
        display_name: str = "eval",
        broadcast_output: Optional[bool] = None,
    ) -> Optional[
        Union[
            Dict[str, Union[Any, Dict[str, float]]],
            Dict[str, Dict[str, Union[Any, Dict[str, float]]]],
        ]
    ]:
        """Run evaluation and return metrics and collected logits (i.e., scores).

        The intermediate embeddings are written to a temporary cache. If the user does not explicitly
        ask to cache the embeddings, the temporary cache is deleted before returning from this function.

        Args:
            eval_dataset (Optional[): dataset to evaluate (if not provided, use ``RetrievalEvaluator.eval_dataset``)
            logit_collector (Optional[Union[Union[ResultHeapq, FastResultHeapq], Dict[str, Union[ResultHeapq, FastResultHeapq]]]]): One or multiple instances
                of ``ResultHeapq`` or ``FastResultHeapq`` that take the scores for each batch
                and keep track of the topk most similar documents for each query.
            cache_pardir (Optional[os.PathLike]): Write the embedding cache files to this
                directory (if not provided, use ``RetrievalEvaluator.cache_pardir``)
            display_name (str): Name to use in console logs and progress bars.
            broadcast_output (Optional[bool]): (only for distributed environments) If true, the output is duplicated
                across all processes (i.e., this method returns identical output in all processes).
                If False, only the main process returns the output and other processes return None.
                Set it to False to save memory on machines with multiple GPUs.

        Returns:
            a mapping with two keys, ``metrics`` and ``logits``. ``metrics`` is a mapping from metric name to metric value.
            ``logits`` is the subset of scores collected by ``logit_collector`` and
            is obtained by calling ``logit_collector.as_qrel_nested_dict``.
            If ``eval_dataset`` is a dict, we return a mapping from keys in ``eval_dataset``
            to the results for the corresponding dataset.
        """
        _name = display_name  # short variable names are cleaner

        if broadcast_output is None:
            broadcast_output = self.args.broadcast_output

        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        if eval_dataset is None:
            msg = "There is no dataset to evaluate."
            raise ValueError(msg)
        if logit_collector is None:
            logit_collector = self.logit_collector
        if cache_pardir is None:
            cache_pardir = self.cache_pardir

        if isinstance(eval_dataset, dict):
            results = dict()
            for ds_name, ds in eval_dataset.items():
                if isinstance(logit_collector, dict):
                    lc = logit_collector[ds_name]
                else:
                    lc = logit_collector
                _out = self._evaluate_one(
                    eval_dataset=ds,
                    logit_collector=lc,
                    cache_pardir=cache_pardir,
                    display_name=f"{_name}-{ds_name}",
                )
                results[ds_name] = _out

        else:
            _results = self._evaluate_one(
                eval_dataset=eval_dataset,
                logit_collector=logit_collector,
                cache_pardir=cache_pardir,
                display_name=_name,
            )
            results = dict(eval=_results)

        if self.is_main_process:
            # reformat the metrics to metrics[ds_name/metric_name] = metric_value
            # This creates a new section for each dataset in wandb dashboard
            metrics_to_log = dict()
            metrics_to_print = dict()
            for ds_name, ds_output in results.items():
                metrics_to_print[ds_name] = ds_output["metrics"]
                for mname, mval in ds_output["metrics"].items():
                    metrics_to_log[f"{ds_name}/{mname}"] = mval

            self._print(json.dumps(metrics_to_print, indent=2))
            self.log_metrics(metrics_to_log)

            if self.args.output_dir is not None:
                logger.info("Writing evaluation results to disk.")
                for ds_name, ds_results in results.items():
                    dst_dir = Path(self.args.output_dir, "evaluation_results")
                    if isinstance(eval_dataset, dict):
                        # only create a subdir for dataset if eval_dataset is a dict and has a corresponding key for each dataset
                        dst_dir = dst_dir.joinpath(ds_name)
                    self.write_json(
                        obj=ds_results["metrics"],
                        path=dst_dir.joinpath("metrics.json"),
                        indent=2,
                    )
                    if self.args.save_eval_topk_logits:
                        for res_k in ds_results["logits"].keys():
                            res_v = ds_results["logits"][res_k]

                            res_local_id = defaultdict(dict)
                            for g_qid, _topdocs in res_v.items():
                                _qid, _ = g_qid.rsplit("_", maxsplit=1)
                                for g_docid, _s in _topdocs.items():
                                    _docid, _ = g_docid.rsplit("_", maxsplit=1)
                                    if _docid in res_local_id[_qid]:
                                        raise RuntimeError(
                                            "Found two records with the same ID. Try setting `save_eval_topk_logits` to `False`"
                                        )
                                    res_local_id[_qid][_docid] = _s
                            res_v = dict(res_local_id)
                            del res_local_id
                            if self.args.output_qrel_format == "tsv":
                                res_path = dst_dir.joinpath(f"logits_{res_k}.tsv")
                            elif self.args.output_qrel_format == "grouped":
                                res_path = dst_dir.joinpath(f"logits_{res_k}.jsonl")
                            else:
                                raise ValueError

                            with file_utils.atomic_write(
                                file=res_path, root="parent"
                            ) as tfile:
                                file_utils.write_qrels(
                                    qrels=res_v,
                                    filepath=tfile,
                                    format=self.args.output_qrel_format,
                                )

            if not isinstance(eval_dataset, dict):
                # If original dataset was not a dict, you should not return the results as dict
                results = list(results.values())[0]
        else:
            results = None

        self.barrier()
        if broadcast_output:
            # copy the output from process 0 to all other processes
            logger.info("Broadcast output to all processes.")
            results = self.broadcast_obj(obj=results, src=0)
        return results

    def _mine_hard_negatives_one(
        self,
        eval_dataset: MultiLevelDataset,
        logit_collector: Union[ResultHeapq, FastResultHeapq],
        cache_pardir: os.PathLike,
        display_name: str,
        num_negs: Optional[int] = None,
    ) -> Optional[List[Dict[str, Union[os.PathLike, Dict[str, Dict[str, float]]]]]]:
        """Mine hard negatives for one eval dataset.

        .. note::

            To save memory in distributed environments, this method only returns the output in process with ``rank == 0`` and returns ``None`` in other processes.

        See docstring for :meth:`mine_hard_negatives` for details.
        """
        logger.info(f"Mining hard negatives for: {display_name}")

        if self.args.no_annot_in_mined_hn:
            gt_qrel = eval_dataset.get_qrel_nested_dict(return_global_ids=True)
        else:
            # pretending to not have any annotations.
            gt_qrel = dict()

        max_gt_docs_per_query = None
        if len(gt_qrel) != 0:
            # maximum number of annotated docs per query
            max_gt_docs_per_query = max([len(item) for item in gt_qrel.values()])

        # free up memory for NN search. We recreate it again later
        del gt_qrel

        if num_negs is None:
            num_negs = logit_collector.topk

        # We want to choose num_negs unannotated docs for each query
        # Instead of excluding annotated docs during NN search (which is slow),
        # We choose num_negs + max_gt_docs_per_query documents for each query to make sure
        # at least num_negs of chosen documents are unannotated
        orig_topk = logit_collector.topk
        if max_gt_docs_per_query is None:
            logit_collector.topk = num_negs
        else:
            logit_collector.topk = num_negs + max_gt_docs_per_query
        # after changing topk, clear the internal state of logit_collector to make sure
        # it holds valid data at all times
        logit_collector.reset_state()

        query_datasets, corpus_datasets = eval_dataset.get_encoding_datasets(
            encoding_cache_pardir=cache_pardir, load_cache_on_init=False
        )
        # unique global file ids that extend query and corpus IDs. See EncodingDataset.__getitem__ for details
        global_id_to_filepath = dict()
        for eds in query_datasets + corpus_datasets:
            global_id_to_filepath[eds.global_id_suffix[1:]] = eds.filepath

        # currently, we can only convert global IDs to local IDs if the ID suffix does not
        # contains '_' character
        if not all(["_" not in _id for _id in global_id_to_filepath.keys()]):
            msg = (
                "At least one of the global file IDs contain an '_' character, "
                "which is not supported at this point."
            )
            raise RuntimeError(msg)

        search_results = self.nearest_neighbor_search(
            query_dataset=query_datasets,
            corpus_dataset=corpus_datasets,
            logit_collector=logit_collector,
            cache_pardir=cache_pardir,
            display_name=display_name,
        )
        # free up some memory
        del query_datasets
        del corpus_datasets
        del eds
        gc.collect()

        if not self.is_main_process:
            # to save memory, we only calculate the output and return it in the main process.
            # And avoid duplicate computation in other processes
            return None
        search_results = search_results["topk_docs"]
        # reverse the topk property of the logit_collector and reset its state to make sure
        # this object remains valid for others to use
        logit_collector.topk = orig_topk
        logit_collector.reset_state()
        gc.collect()

        if self.args.no_annot_in_mined_hn:
            gt_qrel = eval_dataset.get_qrel_nested_dict(return_global_ids=True)
        else:
            # pretending to not have any annotations.
            gt_qrel = dict()

        # mined_qrels[query_file_id][corpus_file_id] contains a subset of mined qrel triplets
        # that their query and documents come from files with 'query_file_id' and 'corpus_file_id' global IDs
        # mined_qrels[query_file_id][corpus_file_id][qid][docid] = similarity_score(qid, docid)
        mined_qrels = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        for g_qid, docdata in self.pbar(
            search_results.items(),
            desc="Filter annotated docs from NN search",
            disable=not logging_conf.is_debug(),
        ):
            if len(gt_qrel) == 0:
                # There is no information about annotated docs for any of the queries
                # So all documents are used during HN mining
                _docdata = docdata.items()
            elif g_qid not in gt_qrel:
                # We have annotation information in general but we don't have any
                # information about annotated docs for this specific querys. So we skip it
                continue
            else:
                # exclude annotated docs
                _docdata = [i for i in docdata.items() if i[0] not in gt_qrel[g_qid]]
            _docdata = sorted(_docdata, key=lambda x: (x[1], x[0]), reverse=True)
            _docdata = _docdata[:num_negs]
            # convert global IDs to local IDs
            qid, query_f_id = g_qid.rsplit("_", maxsplit=1)
            for g_docid, docscore in _docdata:
                # convert global IDs to local IDs
                docid, corpus_f_id = g_docid.rsplit("_", maxsplit=1)
                mined_qrels[query_f_id][corpus_f_id][qid][docid] = docscore

        del gt_qrel  # free memory

        logger.debug(
            "Changing mined qrel format to a list of nested qrel dicts with one entry per query-corpus file pair."
        )
        mined_qrels_list = list()
        for query_f_id, query_f_data in mined_qrels.items():
            for corpus_f_id, query_corpus_f_qrels in query_f_data.items():
                mined_qrels_list.append(
                    {
                        "query_file": global_id_to_filepath[query_f_id],
                        "corpus_file": global_id_to_filepath[corpus_f_id],
                        "qrel": dict(query_corpus_f_qrels),
                    }
                )
        if self.args.merge_mined_qrels and len(mined_qrels_list) > 1:
            merged_qrels = defaultdict(dict)
            for pair_data in mined_qrels_list:
                for qid, topdocs in pair_data["qrel"].items():
                    for docid, score in topdocs.items():
                        if docid in merged_qrels[qid]:
                            raise RuntimeError(
                                "Found two records with the same ID. Try setting `merge_mined_qrels` to `False`"
                            )
                        merged_qrels[qid][docid] = score
            mined_qrels_list = [
                {"query_file": "all", "corpus_file": "all", "qrel": dict(merged_qrels)}
            ]

        return mined_qrels_list

    @file_utils.call_cleanup("remove_temp_cache_pardir")
    def mine_hard_negatives(
        self,
        eval_dataset: Optional[
            Union[MultiLevelDataset, Dict[str, MultiLevelDataset]]
        ] = None,
        query_filepath: Optional[Union[os.PathLike, List[os.PathLike]]] = None,
        corpus_filepath: Optional[Union[os.PathLike, List[os.PathLike]]] = None,
        logit_collector: Optional[
            Union[
                Union[ResultHeapq, FastResultHeapq],
                Dict[str, Union[ResultHeapq, FastResultHeapq]],
            ]
        ] = None,
        cache_pardir: Optional[os.PathLike] = None,
        display_name: str = "eval",
        num_negs: Optional[int] = None,
        broadcast_output: Optional[bool] = None,
    ) -> Optional[
        Union[
            List[Dict[str, Union[os.PathLike, Dict[str, Dict[str, float]]]]],
            Dict[str, List[Dict[str, Union[os.PathLike, Dict[str, Dict[str, float]]]]]],
        ]
    ]:
        """Mine the most similar documents for each query as hard negatives.

        Retrieves the topk most similar documents for each query as hard negatives.

        In the case that the resulting ``eval_dataset`` object (whether given or created
        from ``query_filepath``, and ``corpus_filepath``) contains a valid set of qrel triplets,
        queries that have no corresponding qrel triplet are ignored. I.e., no documents are mined for them
        and they are not included in the returned results.

        Mined hard negatives are returned in nested qrel format. But, instead of
        returning one qrel object, it creates one qrel object for each pair of query
        and corpus files (this allows us to read queries and documents from multiple
        files with potentially none-unique IDs across files). It returns a list of dicts, each corresponding to a
        pair of query and corpus files::

            [
                {
                    'query_file': 'path to file that contains the corresponding queries.',
                    'corpus_file': 'path to file that contains the corresponding documents.',
                    'qrel': '''a subset of mined hard negatives that contain only queries and documents
                        from 'query_file' and 'corpus_file'. it is in nested qrel format (i.e., qrel[qid][docid]=similarity(qid, docid)).'''
                }

                ...
            ]

        If ``args.output_dir`` is provided, mined hard negatives are also written to disk in grouped qrel format
        in a json lines file.

        Args:
            eval_dataset: dataset to mine hard negatives for.
            query_filepath: file to read queries from.
            corpus_filepath: file to read documents from.
            logit_collector: Instance of ``ResultHeapq`` or ``FastResultHeapq`` that takes the scores
                for each batch and keeps the topk most similar documents and their scores for each query.
            cache_pardir: Write the embedding cache files to this directory.
            display_name: Name to use in console logs and progress bars.
            num_negs: number of hard negatives to mine per query. If not provided, use the value of ``args.search_topk``.
            broadcast_output (Optional[bool]): (only for distributed environments) If true, the output is duplicated
                across all processes (i.e., this method returns identical output in all processes).
                If false, only the main process returns the output and other processes return ``None``.
                Set it to false to save memory on machines with multiple GPUs.

        Returns:
            A list of mined hard negatives with one entry for each contributing pair of query
            and corpus files. See extended method docstring for details. If ``eval_dataset`` is a dict,
            we return a mapping from keys in ``eval_dataset`` to the described results for
            the corresponding dataset.
        """
        _name = display_name  # short variable names are cleaner

        if (query_filepath is None) != (corpus_filepath is None):
            msg = (
                "You should provide either both or none of 'query_filepath' and 'corpus_filepath'."
                f" Got: query_filepath={query_filepath}\n"
                f"corpus_filepath={corpus_filepath}"
            )
            raise ValueError(msg)

        if eval_dataset is not None and query_filepath is not None:
            msg = "You should provide either eval_dataset or query/corpus file paths. But, not both."
            raise ValueError(msg)

        if query_filepath is not None or corpus_filepath is not None:
            msg = (
                "Mining hard negatives from corpus and query files is not yet supported."
                " Pass an instance of 'MultiLevelDataset' to `eval_dataset` instead."
            )
            raise NotImplementedError(msg)

        if broadcast_output is None:
            broadcast_output = self.args.broadcast_output

        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        if eval_dataset is None:
            msg = "There is no dataset for hard negative mining."
            raise ValueError(msg)
        if logit_collector is None:
            logit_collector = self.logit_collector
        if cache_pardir is None:
            cache_pardir = self.cache_pardir

        if isinstance(eval_dataset, dict):
            results = dict()
            for ds_name, ds in eval_dataset.items():
                if isinstance(logit_collector, dict):
                    lc = logit_collector[ds_name]
                else:
                    lc = logit_collector
                _out = self._mine_hard_negatives_one(
                    eval_dataset=ds,
                    logit_collector=lc,
                    cache_pardir=cache_pardir,
                    display_name=f"{_name}-{ds_name}",
                    num_negs=num_negs,
                )
                results[ds_name] = _out
        else:
            _results = self._mine_hard_negatives_one(
                eval_dataset=eval_dataset,
                logit_collector=logit_collector,
                cache_pardir=cache_pardir,
                display_name=_name,
                num_negs=num_negs,
            )

            results = dict(eval=_results)

        if self.is_main_process:
            if self.args.output_dir is not None:
                logger.info("Writing mined qrel triplets to disk.")
                for ds_name, ds_results in results.items():
                    for pair_idx, pair_data in enumerate(ds_results):
                        dst_dir = Path(self.args.output_dir, "mined_hard_negatives")
                        if isinstance(eval_dataset, dict):
                            # only add ds_name if eval_dataset is a dict and has a corresponding key and NOT if we set ds_name to eval ourselves above
                            dst_dir = dst_dir / ds_name
                        if not self.args.merge_mined_qrels:
                            dst_dir = dst_dir / f"query_corpus_pair_{pair_idx}"
                            pair_metadata = {
                                "query_file": str(pair_data["query_file"]),
                                "query_realpath": str(
                                    file_utils.realpath(pair_data["query_file"])
                                ),
                                "corpus_file": str(pair_data["corpus_file"]),
                                "corpus_realpath": str(
                                    file_utils.realpath(pair_data["corpus_file"])
                                ),
                            }
                            self.write_json(
                                obj=pair_metadata,
                                path=dst_dir.joinpath("file_pair_metadata.json"),
                            )
                        if self.args.output_qrel_format == "tsv":
                            res_path = dst_dir.joinpath("mined_qrels.tsv")
                        elif self.args.output_qrel_format == "grouped":
                            res_path = dst_dir.joinpath("mined_qrels.jsonl")
                        else:
                            raise ValueError

                        with file_utils.atomic_write(
                            file=res_path, root="parent"
                        ) as tfile:
                            file_utils.write_qrels(
                                qrels=pair_data["qrel"],
                                filepath=tfile,
                                format=self.args.output_qrel_format,
                            )

            if not isinstance(eval_dataset, dict):
                # If original dataset was not a dict, you should not return the results as dict
                results = list(results.values())[0]
        else:
            results = None

        self.barrier()
        if broadcast_output:
            # copy the output from process 0 to all other processes
            logger.info("Broadcast output to all processes.")
            results = self.broadcast_obj(obj=results, src=0)
        return results

    def remove_temp_cache_pardir(self) -> None:
        """Delete cache pardir and all its content if it was supposed to be temporary.

        Does not raise exception on failure.
        """
        if not self.args.cleanup_temp_artifacts:
            return

        # Make sure all processes are done before attempting to delete the temp dir.
        self.barrier()
        if self.is_main_process:
            if self.cache_pardir is None:
                # There is nothing to remove
                return
            if not Path(self.cache_pardir).exists():
                # There is nothing to remove
                return
            if self.args.encoding_cache_dir is not None and Path(
                self.args.encoding_cache_dir
            ).samefile(Path(self.cache_pardir)):
                # It is not a temporary cache created by evaluator
                return

            # It is a temp dir and should be removed
            logger.debug(
                f"Attempting to delete the temporary cache dir: {rpath(self.cache_pardir)}"
            )
            file_utils.safe_remove_path(self.cache_pardir)
