Inference
==================

.. role:: raw-html(raw)
    :format: html

.. |RetrievalTrainingArguments| replace:: :py:class:`~trove.trainer.RetrievalTrainingArguments`
.. |ModelArguments| replace:: :py:class:`~trove.modeling.model_args.ModelArguments`
.. |BiEncoderRetriever| replace:: :py:class:`~trove.modeling.retriever_biencoder.BiEncoderRetriever`
.. |DataArguments| replace:: :py:class:`~trove.data.data_args.DataArguments`
.. |MaterializedQRelConfig| replace:: :py:class:`~trove.containers.materialized_qrel_config.MaterializedQRelConfig`
.. |BinaryDataset| replace:: :py:class:`~trove.data.ir_dataset_binary.BinaryDataset`
.. |MultiLevelDataset| replace:: :py:class:`~trove.data.ir_dataset_multilevel.MultiLevelDataset`
.. |RetrievalCollator| replace:: :py:class:`~trove.data.collator.RetrievalCollator`
.. |RetrievalTrainer| replace:: :py:class:`~trove.trainer.RetrievalTrainer`
.. |IRMetrics| replace:: :py:class:`~trove.evaluation.metrics.IRMetrics`
.. |EvaluationArguments| replace:: :py:class:`~trove.evaluation.evaluation_args.EvaluationArguments`
.. |FastResultHeapq| replace:: :py:class:`~trove.containers.result_heapq_fast.FastResultHeapq`
.. |RetrievalEvaluator| replace:: :py:class:`~trove.evaluation.evaluator.RetrievalEvaluator`


Trove streamlines common inference tasks in IR pipelines.
We can easily

- evaluate retrievers and report IR metrics
- mine hard negatives
- run distributed inference (multi GPU/Node) with the same code

Evaluation
---------------------

Inference workflow is very similar to training, which we described in detail in :doc:`training` section.
Here, we walk through the entire evaluation example and explain the steps that differ from training.

Eval Arguments
~~~~~~~~~~~~~~~~~~~~~~

First, we creat an instance of |EvaluationArguments|, which provides various options to control the evaluation process.
Note that |EvaluationArguments| is a subclass of ``transformers.TrainingArguments`` and we reuse some of its options during evaluation but ignore most of them.
See |EvaluationArguments| documentation for details.

.. code-block:: python

    from trove import EvaluationArguments

    eval_args = EvaluationArguments(
        output_dir="./eval_results",
        encoding_cache_dir="./model_encoding_cache_root", # Only needed if you want to keep the cached embeddings
        broadcast_output=False, # save memory
        report_to=["wandb"], # report metrics to wandb
    )

.. tip::

    If you have a reasonable disk performance, you can speed up the process by computing all the embeddings before starting the nearest neighbor search.
    For best performance, set ``precompute_corpus_embs=True`` to precompute the embeddings.
    Maximize ``per_device_eval_batch_size`` value based on your GPU memory (this is used for calculating the embeddings).
    Set ``per_device_matmul_batch_size`` to a very large value (something like ``40,960``).
    This is the batch size used for ``matmul`` operation when calculating the similarity between precomputed embeddings.

    Background info: Trove uses a GPU based implementation of Heapq (|FastResultHeapq|) to keep track of top-k documents.
    |FastResultHeapq| works best with very large batch sizes for nearest neighbor search.


Loading the Model
~~~~~~~~~~~~~~~~~~~~~~

Similar to training.
See :doc:`training` for details.

.. code-block:: python

    from trove import ModelArguments, BiEncoderRetriever
    from transformers import AutoTokenizer

    model_args = ModelArguments(
        model_name_or_path="facebook/contriever",
        encoder_class="default",
        pooling="mean",
        normalize=False,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    model = BiEncoderRetriever.from_model_args(args=model_args)

Creating Evaluation Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This part is also very similar to the training workflow with minor differences.

.. code-block:: python

    from trove import DataArguments, MaterializedQRelConfig, MultiLevelDataset, RetrievalCollator

    data_args = DataArguments(
        dataset_name="msmarco",
        query_max_len=32,
        passage_max_len=128,
    )
    mqrel_conf = MaterializedQRelConfig(
        qrel_path="test_qrel.tsv",
        corpus_path="corpus.jsonl",
        corpus_cache="corpus_emb_cache.arrow"
        query_path="queries.jsonl",
        query_cache="queries_emb_cache.arrow"
    )
    dataset = MultiLevelDataset(
        data_args=data_args,
        format_query=model.format_query,
        format_passage=model.format_passage,
        qrel_config=mqrel_conf,
        num_proc=8,
    )
    data_collator = RetrievalCollator(
        data_args=data_args,
        tokenizer=tokenizer,
        append_eos=model.append_eos_token,
    )

Cache files are created inside ``eval_args.encoding_cache_dir`` directory.
By defult, the name of each embedding cache file is created using the hash of the corresponding input file.
You can change this by setting the value of ``corpus_cache`` and ``query_cache``.

.. attention::

    Only |MultiLevelDataset| can be used for evaluation (and not |BinaryDataset|).


Evaluator
~~~~~~~~~~~~~~~~~~~~~~

Finally, we create an instance of |RetrievalEvaluator| which takes care of all the steps needed for evaluation.
For example, it calculates the embeddings, runs an exhaustive nearest neighbor search, calculates IR metrics, log metrics to wandb, etc.

.. code-block:: python

    from trove import RetrievalEvaluator

    evaluator = RetrievalEvaluator(
        args=eval_args,
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        eval_dataset=dataset,
    )
    evaluator.evaluate()


Hard Negative Mining
-------------------------

The process is identical to evaluation steps explained above with some small changes.


Update |EvaluationArguments|:


.. code-block:: python

    eval_args = EvaluationArguments(
        output_dir="./hn_mining_results",
        encoding_cache_dir="./model_encoding_cache_root", # Only needed if you want to cache the embeddings.
        broadcast_output=False, # save memory
        search_topk=15, # mine 15 negatives for each query
        no_annot_in_mined_hn=True,
        merge_mined_qrels=True,
    )

To exclude the annotated positives from hard negative mining results, we set ``no_annot_in_mined_hn=True``.
However, this excludes all annotated documents.
Even documents that are labeled as negatives are excluded from the results.
To include the annotated negatives as potential hard negatives, exclude the negative annotations when creating the dataset.

For example, when labels are binary (only 0 and 1), we can exclude the negative annotations from the dataset like this:

.. code-block:: python

    mqrel_conf = MaterializedQRelConfig(
        min_score=1, # only include documents with labels >= 1
        qrel_path="train_qrel.tsv",
        ...
    )

Finally, call the method for hard negative mining instead of evaluation:

 .. code-block:: python

    evaluator.mine_hard_negatives()

This is all that you need to change.
The evaluator creates a qrel file with scores for the top-k retrieved documents for each query in the output directory.

Encoding
-------------------------

You can also use |RetrievalEvaluator| to just encode the queries and documents and cache the embeddings without any further processing.
Everything remains the same as hard negative mining or evaluation except the last method you call.

.. code-block:: python

    evaluator.encode()

**Note**: to reuse these cached embeddings later, specify the same ``encoding_cache_dir`` in |EvaluationArguments|.

.. note::

    To make it more conveniet and efficient, the embedding cache is shared between these steps.
    For example, if you evaluate a model and cache the embeddings, then you can reuse the embeddings to mine hard negatives from the same files using the same model.
    And the good thing is that you don't need to explicitly keep track of the cache for each input file.
    As long as the cache files are saved in the same directory, Trove figures out which cache file to use for each input file.


Distributed Inference
---------------------

You can run any of the above tasks in a distributed environment (multi GPU/node).
You just need to launch your script with a distributed launcher.

.. code-block:: bash

    deepspeed --include localhost:0,1 my_script.py {script arguments}

Note that deepspeed is just used as a distributed launcher like ``accelerate``.
|RetrievalEvaluator| does not support integration with deepspeed.
