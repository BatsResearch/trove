Training
==============


.. role:: raw-html(raw)
    :format: html


.. py:currentmodule:: trove


.. |RetrievalTrainingArguments| replace:: :py:class:`~trove.trainer.RetrievalTrainingArguments`
.. |ModelArguments| replace:: :py:class:`~trove.modeling.model_args.ModelArguments`
.. |BiEncoderRetriever| replace:: :py:class:`~trove.modeling.retriever_biencoder.BiEncoderRetriever`
.. |DataArguments| replace:: :py:class:`~trove.data.data_args.DataArguments`
.. |MaterializedQRelConfig| replace:: :py:class:`~trove.containers.materialized_qrel_config.MaterializedQRelConfig`
.. |PretrainedEncoder| replace:: :py:class:`~trove.modeling.pretrained_encoder.PretrainedEncoder`
.. |BinaryDataset| replace:: :py:class:`~trove.data.ir_dataset_binary.BinaryDataset`
.. |MultiLevelDataset| replace:: :py:class:`~trove.data.ir_dataset_multilevel.MultiLevelDataset`
.. |RetrievalCollator| replace:: :py:class:`~trove.data.collator.RetrievalCollator`
.. |RetrievalTrainer| replace:: :py:class:`~trove.trainer.RetrievalTrainer`
.. |IRMetrics| replace:: :py:class:`~trove.evaluation.metrics.IRMetrics`



Trove uses huggingface transformers for training.
You should just use ``trove.RetrievalTrainer`` instead of ``transformers.Trainer``, which makes small modifications to allow saving the checkpoints for ``PretrainedRetriever`` subclasses.
It also scales up the loss value for aggregation in distributed environments (the effective loss value remains the same).
Everything else remains the same as ``transformers.Trainer``.

Workflow
---------------------

Here we explain the workflow of Trove for training.
The example explained here is roughly equal to `this script <https://github.com/BatsResearch/trove/blob/main/examples/train_simple.py>`_.

Loading the Model
~~~~~~~~~~~~~~~~~~~~~~

First, we create an instance of |RetrievalTrainingArguments|. This is identical to ``transformers.TrainingArguments`` and just adds one extra option (``trove_logging_mode``) to control the logging mode for Trove.
You have access to all arguments in `transformers.TrainingArguments <https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments>`_ like learning rate, save frequency, etc.

.. code-block:: python

    from trove import RetrievalTrainingArguments

    train_args = RetrievalTrainingArguments(output_dir="./my_model", learning_rate=1e-5, ...)

Next, we create an instance of |ModelArguments| which determines how the model should be loaded and used.

.. code-block:: python

    from trove import ModelArguments

    model_args = ModelArguments(
        model_name_or_path="facebook/contriever",
        encoder_class="default",
        pooling="mean",
        normalize=False,
        loss="infonce"
    )

We need to specify which wrapper should be used to load the encoder.
``encoder_class="default"`` means that the model checkpoint (``model_name_or_path``) should be loaded with a subclass of |PretrainedEncoder| that its name (or alias) is ``"default"``.

.. tip::

    If the specified encoder wrapper supports it, we can use |ModelArguments| to ask the wrapper to quantize the encoder or add LORA adapters using options like ``use_peft`` and ``load_in_4bit``.
    The default wrapper supports both LORA and quantization.

Similarly, ``loss="infonce"`` specifies the name (or alias) of the loss function that should be instantiated.

.. tip::

    Use ``trove.RetrievalLoss.available_losses()`` to see the name of all available loss functions.

See :doc:`Modeling <modeling>` for how you can add custom encoder wrappers and loss functions.



Next, we create a bi-encoder retriever (|BiEncoderRetriever|) using this config.
The retriever instantiates the encoder and loss function based on the given ``model_args``.

.. code-block:: python

   from trove import BiEncoderRetriever

   model = BiEncoderRetriever.from_model_args(args=model_args)


Creating Training Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, we create an instance of |DataArguments| to specify how the data should be processed.

The ``dataset_name`` is helpful if your encoder expects the inputs to be processed differently based on the dataset (e.g., use different task instructions for each dataset).
``group_size`` is the number of documents used for each query.
In this example, we will create a binary dataset, which means we will have one positive and 15 negatives for each query.

.. code-block:: python

   from trove import DataArguments

   data_args = DataArguments(
        dataset_name="msmarco",
        group_size=16,
        query_max_len=32,
        passage_max_len=128
    )


Next, we create two instances of |MaterializedQRelConfig| for negatives and positives to specify where to find the data and how to load and process it.

.. code-block:: python

    from trove import MaterializedQRelConfig

    pos_conf = MaterializedQRelConfig(
        qrel_path="train_qrel.tsv",
        corpus_path="corpus.jsonl",
        query_path="queries.jsonl",
        min_score=1,
    )
    neg_conf = MaterializedQRelConfig(
        qrel_path="train_qrel.tsv",
        corpus_path="corpus.jsonl",
        query_path="queries.jsonl",
        max_score=1,
    )

Let's consider positives first.
The above snippet says that queries should be loaded from the ``queries.jsonl`` file and documents should be loaded from ``corpus.jsonl`` file.
And, the annotations should be loaded from ``train_qrel.tsv`` file.
Importantly, we filter the documents and do not use all the annotated documents as positives.
But, we restrict positives to only documents that their label is greater than or equal to one (``min_score=1``).

Similarly for negatives, we load the query, corpus, and annotations from the same set of files as positives.
But this time, we restrict negatives to only documents that their label is less than one (``max_score=1``), effectively anything with label zero.

Note that you can also use a list of filenames instead of a single file and the results are merged.


.. tip::

    You can apply more complex data processing pipelines like filtering with arbitrary functions, transforming the scores, etc.
    See :doc:`data` for more information.


Now, we creae a binary training dataset (|BinaryDataset|) using these negative and positive documents.

.. code-block:: python

    from trove import BinaryDataset

    dataset = BinaryDataset(
        data_args=data_args,
        positive_configs=pos_conf,
        negative_configs=neg_conf,
        format_query=model.format_query,
        format_passage=model.format_passage,
    )


The ``format_query`` and ``format_passage`` methods of the model take a raw query and passage and apply whatever processing that the encoder expects.
For example, for models like E5-mistral, these functions are expected to add a task instruction to the query.

.. tip::

    You can use a list of config objects for each of the ``positive_configs`` and ``negative_configs`` arguments to create more complex data pipelines.
    This allows you to combine your positives and negatives each from multiple sources.
    You can even process each data source differently before merging.

.. tip::

    ``BinaryDataset`` is suitable for training with binary relevance labels using InfoNCE loss.
    If you want to train with multiple levels of relevance (e.g., labels from ``{0, 1, 2, 3}``), you need
    to use |MultiLevelDataset| instead of |BinaryDataset|.
    The process is very similar.
    You just need one set of ``MaterializedQRelConfig`` objects instead of the two sets expected by the binary dataset for positives and negatives.


We also create a data collator (|RetrievalCollator|) that takes care of processes like tokenization, truncation, padding, etc.

.. code-block:: python

    from transformers import AutoTokenizer
    from trove import RetrievalCollator

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    data_collator = RetrievalCollator(
        data_args=data_args,
        tokenizer=tokenizer,
        append_eos=model.append_eos_token,
    )

If ``append_eos`` is set to ``True``, the collator makes sure all input sequences end with an ``eos`` token.
This is helpful when using last-token pooling.
Ideally, encoders should specify if they expect an ``eos`` token or not. So, we can use ``model.append_eos_token`` to correctly config the data pipeline without any manual effort.

Trainer
~~~~~~~~~~~~~~~~~~~~~~

Finally, we create an instance of |RetrievalTrainer| for training.
|RetrievalTrainer| is almost identical to ``transformers.Trainer`` with very small changes, which does not impact how it is used.
In users' training scripts, everything is the same as ``transformers.Trainer``.


.. code-block:: python

    from trove import RetrievalTrainer

    trainer = RetrievalTrainer(
        args=train_args,
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model()


Distributed Training
---------------------


Trove is fully compatible with huggingface transformers ecosystem.
So, you can just launch your script using a distributed launcher and that is all you need to do for distributed training across multiple nodes and GPUs.

Similarly, you can use deepspeed just as you do with any transformers training script.

.. code-block:: bash

    deepspeed --include localhost:0,1 my_script.py \
        --deepspeed 'my_deepspeed_config.json'
        # rest of your script arguments


.. note::

   Trove retrievers automatically collect in-batch negatives from across devices and nodes.


IR Metrics During Training
-----------------------------

During training, Trove can report IR metrics like nDCG for a dev set.
Since calculating exact metrics multiple times on the entire corpus is too expensive, we choose to approximate IR metrics on a small subset of annotated documents for each query.
It is sort of similar to a reranking task.
For instance, given a dev set that provides a reasonable number of annotations (~100) per query, we can rank `only` these annotated documents (and not the entire corpus) for each query and calculate IR metrics based on that.

Most of the above code remains the same except for a few changes.
First we need to update the training arguments.

.. code-block:: python

    train_args = RetrievalTrainingArguments(
        ...
        batch_eval_metrics=True,
        label_names=["label"],
        eval_strategy="steps",
        eval_steps=1000,
    )

Next you need to create an evaluation dataset.

.. code-block:: python

    eval_mqrel = MaterializedQRelConfig(
        qrel_path="dev_qrel.tsv",
        corpus_path="corpus.jsonl",
        query_path="queries.jsonl",
    )

.. tip::

    Often such annotated dev sets are not available and we only have a few positives for each query.
    In these cases, we can mine a limited number of negatives (~100) for each query in the dev set.
    We then combine these mined negatives with annotated positives to create a dev set for approximating IR metrics during training.

    .. code-block:: python

        eval_mqrel = [
            MaterializedQRelConfig(
                qrel_path="dev_qrel_positives.tsv",
                corpus_path="corpus.jsonl",
                query_path="queries.jsonl",
                score_transform=1
            ),
            MaterializedQRelConfig(
                qrel_path="dev_mined_negs.tsv",
                corpus_path="corpus.jsonl",
                query_path="queries.jsonl",
                score_transform=0
            )
        ]

For evaluations, we must use |MultiLevelDataset| even if our labels are binary.

.. code-block:: python

    arg_overrides = {"group_size": 100, "passage_selection_strategy": "most_relevant"}
    eval_dataset = MultiLevelDataset(
        data_args=data_args,
        format_query=model.format_query,
        format_passage=model.format_passage,
        qrel_config=eval_mqrel,
        data_args_overrides=arg_overrides,
        num_proc=8,
    )

We reuse the same ``data_args`` object that we used for the training dataset; but we override the value of some attributes with ``arg_overrides``.

To calculate approximate metrics, Trove expects all queries to have the same number of documents (for easier batching).
So, we set ``"group_size":100`` to make sure all queries have 100 annotated documents.
If a query has more than 100 documents, ``MultiLevelDataset`` uses a subset of annotated documents.
To make sure the positive documents are included in the subst, we set ``"passage_selection_strategy": "most_relevant"``.
If a query has fewer than 100 annotated documents, ``MultiLevelDataset`` duplicates some documents.

We also need to create a stateful callback function (|IRMetrics|) to compute the metrics for each batch of eval data.

.. code-block:: python

    from trove import IRMetrics

    # k_values are cutoff values for IR metrics
    metric_callback = IRMetrics(k_values=[10, 100])


Finally, we add a few extra arguments when instantiating and using the trainer.

.. code-block:: python

    trainer = RetrievalTrainer(
        ...
        eval_dataset=eval_dataset,
        compute_metrics=metric_callback,
    )
    trainer.train(ignore_keys_for_eval=["query", "passage"])
