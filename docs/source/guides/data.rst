Data
==================

.. role:: raw-html(raw)
    :format: html


.. |MaterializedQRel| replace:: :py:class:`~trove.containers.materialized_qrel.MaterializedQRel`
.. |MaterializedQRelConfig| replace:: :py:class:`~trove.containers.materialized_qrel_config.MaterializedQRelConfig`
.. |MultiLevelDataset| replace:: :py:class:`~trove.data.ir_dataset_multilevel.MultiLevelDataset`
.. |BinaryDataset| replace:: :py:class:`~trove.data.ir_dataset_binary.BinaryDataset`
.. |RetrievalCollator| replace:: :py:class:`~trove.data.collator.RetrievalCollator`
.. |InfoNCELoss| replace:: :py:class:`~trove.modeling.losses.InfoNCELoss`
.. |DataArguments| replace:: :py:class:`~trove.data.data_args.DataArguments`
.. |RetrievalTrainer| replace:: :py:class:`~trove.trainer.RetrievalTrainer`
.. |RetrievalEvaluator| replace:: :py:class:`~trove.evaluation.evaluator.RetrievalEvaluator`


Trove allows to filter, select, transform and even combine multiple data sources easily and on-the-fly.
We use tools and technologies like `Apache Arrow <https://arrow.apache.org/docs/python/index.html>`_, `Polars <https://pola.rs/>`_, etc. to do this efficiently with minimal memory consumption.
Trove also caches the intermediate results to further speedup the process when you reuse the same data pipeline.

The good thing is that you just need to specify how to prepare the data and then focus on your experiments, without keeping track of large preprocessed data files for each experiment.
This is very helpful for quickly trying several ideas.
It also helps with reproducibility: you can keep track of your data specifications using a version control system like git and ignore the large data files for each experiment.
Trove creates the same data every time you run your code.


Trove Datasets
---------------------

There are two main dataset classes in Trove, |MultiLevelDataset| and |BinaryDataset|.
We use |MultiLevelDataset| for training with graduated relevance labels (e.g., ``{0, 1, 2, 3}``).
We also have to use |MultiLevelDataset| for any evaluation, encoding, and hard negative mining tasks.
On the other hand, |BinaryDataset| is only used for training and only when you have binary relevance labels, i.e., positives and negatives.

.. warning::

    When using Trove's builtin InfoNCE loss (|InfoNCELoss|), you must use an instance of |BinaryDataset| for training.
    See :ref:`modeling-loss-functions` for more details.

Trove datasets are made up of one or more instances of |MaterializedQRel|.
Each |MaterializedQRel| instance contains a collection of queries, documents, and their relation (i.e., annotations).

To save memory, |MaterializedQRel| only works with query and document IDs and loads the actual data (i.e., materializes the records) only when it is needed.
Even then, the data is memory mapped to minimize memory consumption with negligible impact on performance.

..  attention::

    Query IDs in each |MaterializedQRel| instance must be unique.
    If you have multiple query files that use the same ID for different queries, you should load then in separate ``MaterializedQRel`` instances.
    The same is true for document IDs across multiple corpus files.

You never instantiate a |MaterializedQRel| directly.
Instead, you create |MaterializedQRelConfig| instances with your data specifications and pass these config objects to datasets.

In the most basic case, you can create a |MaterializedQRelConfig| instance from input file paths.
Query and corpus files must be in standard JSONL format.
Qrel files are more flexible. You can even load them from custom file formats (see :ref:`custom-qrel-formats`).

You can use a list of paths to combine several files.

.. code-block:: python

    mqrel_args = trove.MaterializedQRelConfig(
        corpus_path=["/path/to/corpus-0-of-2.jsonl", "/path/to/corpus-1-of-2.jsonl"],
        query_path="/path/to/queries/jsonl",
        qrel_path=["train_qrel-0-of-3.tsv", "train_qrel-1-of-3.tsv", "train_qrel-2-of-3.tsv"],
    )
    dataset = trove.MultiLevelDataset(qrel_config=mqrel_args, ...)

.. important::

    The final dataset only includes queries that have at least one annotation in qrel files.
    Queries that show up in query files (``query_path``) but do not have any corresponding record in qrel files (``qrel_path``) are ignored.

You can use this dataset for both training and evaluation.

**Training**: Each item of this dataset is a training instance.

.. code-block:: python

    >>> dataset[0]
    {
        'query': 'what is the fastest animal?',
        'passage': ['the fastest animal is cheetah', 'cheetah runs very fast', 'there are a lot of fast animals', ...],
        'label': [3, 2, 0, ...]
    }

For a training dataset, you can change the number of passages used for each query or the sampling behavior by changing |DataArguments| attributes when creating the dataset.
See |DataArguments| documentation for all options.

.. code-block:: python

    data_args = trove.DataArguments(group_size=8, ...)
    dataset = trove.MultiLevelDataset(args=data_args, ...)


**Evaluation/Inference**: You can also use this dataset for evaluation (or hard negative mining).
After processing the data and combining data from all |MaterializedQRel| instances, it generates a label dictionary in the format expected by `pytrec_eval <https://github.com/cvangysel/pytrec_eval>`_ package.

.. code-block:: python

    >>> dataset.get_qrel_nested_dict()
    {
        'q1': {'d1': 0, 'd2': 1, ...},
        'q2': {'d1': 1, 'd3': 2, ...}
        ...
    }

.. tip::

   For evaluation, you do not need to work with the dataset or its labels directly.
   You can use :class:`~trove.evaluation.evaluator.RetrievalEvaluator` which takes care of all the steps required for evaluation and hard negative mining.
   See :doc:`inference` for details.



Data Processing
---------------------

|MaterializedQRel| instances also provide various data processing functionalities like filtering, selection, transformation, etc.

As mentioned above, Trove mainly works with record IDs and scores (and not the full query/document).
This data is often held as a collection of dictionaries of three items (e.g., ``{'qid': 'q1', 'docid': 'd2', 'score': 1.0}``).
In this guide, we use `triplet` to refer to each of these dictionaries.

You can apply various data processings to these triplets using |MaterializedQRelConfig| options.


Filtering
~~~~~~~~~~~~~~~~~~~~~~

**Filter Individual Triplets**

You can filter ``(qid, docid, score)`` triplets (represented as dictionary instance) in different ways.
You can filter based on maximum and minimum value of ``score`` or define your own custom filtering function.

For example in a dataset with binary labels (only ``0`` and ``1``), you can do the following

.. code-block:: python

    # only keep negatives (docs with label 0)
    mqrel_args = MaterializedQRelConfig(max_score=1 , ...)
    # only keep positives (docs with label 1)
    mqrel_args = MaterializedQRelConfig(min_score=1 , ...)
    # or any arbitrary function
    # only keep triplets that their document ID ends with '_synth'
    mqrel_args = MaterializedQRelConfig(filter_fn=lambda rec: rec["docid"].endswith("_synth") , ...)
    # only keep triplets that their document ID ends with '_synth' and their label is 3
    mqrel_args = MaterializedQRelConfig(filter_fn=lambda rec: rec["docid"].endswith("_synth") and rec["score"] == 3, ...)


**Filter Subset of Triplets for Each Query**

Sometimes, your filtering logic needs to know the label of all the annotated documents for each query.
For example, assume that you have a dataset with multi-level labels (i.e., ``{0, 1, 2, 3}``).
Now, you want to keep the N most relevant annotated documents for each query.
For this you need to have access to all annotated documents for each query at once.

You can either use Trove's predefined logics or define your custom filtering function.
See |MaterializedQRelConfig| for all available options.

.. code-block:: python

    # For each query, choose the 3 annotated docs with largest scores
    mqrel_args = MaterializedQRelConfig(group_top_k=3 , ...)
    # For each query, choose the 3 annotated docs with smallest scores
    mqrel_args = MaterializedQRelConfig(group_bottom_k=3 , ...)
    # For each query, randomly select 3 annotated docs
    mqrel_args = MaterializedQRelConfig(group_random_k=3 , ...)
    # Define a custom function to filter a list of triplets
    mqrel_args = MaterializedQRelConfig(group_filter_fn=lambda recs: [sorted(recs)[0], sorted(recs)[-1]])


Transformation
~~~~~~~~~~~~~~~~~~~~~~


Trove allows you to change the label values on-the-fly.

For example if you are combining a multi-level dataset (with labels ``{0, 1, 2, 3}``) with a binary dataset (with labels ``{0, 1}``),
you need to change all 1s to 3s in the binary dataset before mixing them. Otherwise your positives in binary dataset will be counted as irrelevant with new label ranges.
Another example is if you want to change a multi-level dataset to a binary dataset.

You can do all of these **without** changing your data files at all.
Set the value of ``score_transform`` either to a fixed constant value or a callable that returns the new triplet scores.
See |MaterializedQRelConfig| documentation for more details.


.. code-block:: python

    # Use all documents in this collection as negatives (assign label 0 to all of them)
    mqrel_args = MaterializedQRelConfig(score_transform=0 , ...)
    # convert a multi-level dataset to binary:
    # Map labels {3, 2} to 1 and labels {1, 0} to 0
    mqrel_args = MaterializedQRelConfig(score_transform=lambda rec: 1 if rec['score'] in [3, 2] else 0, ...)


Selection
~~~~~~~~~~~~~~~~~~~~~~

You can select a subset of queries to be included in the dataset on-the-fly.
For example, assume all your hard negative mining results for all splits are in one qrel file.
And you want to only use the training queries and their hard negatives in the dataset (e.g., for training).

You can do this by specifying a file that contains the subset of query IDs that you are interested in.
It can be a ``queries.jsonl`` file or a qrel file (e.g., ``qrel.tsv``).

.. code-block:: python

    mqrel_args = MaterializedQRelConfig(
        ...
        query_path="queries.jsonl",
        qrel_path='hard_negatives_all_splits.tsv',
        query_subset_path='training_queries.jsonl'
        # or select the target subset of query IDs from another qrel file
        # query_subset_path='orig_train_qrels.tsv'
    )

Combining Data Sources
~~~~~~~~~~~~~~~~~~~~~~

We can create a dataset by combining various data sources.
And since data processing is done by |MaterializedQRel| instances themselves, we can process each data source differently before merging them.
When merging multiple collection, if a query exists in several sources, annotations from all these sources are combined.

To demonstrate, assume that we have the following records in our real and synthetic data collections.

.. code-block:: text

    ## format: (query, passage, label)

    ## real data

    foo, real_A, 1
    foo, real_B, 0
    bar, real_C, 1
    bar, real_D, 0

    ## synthetic data

    foo, synth_A, 3
    foo, synth_B, 1
    foo, synth_C, 0
    qux, synth_D, 3
    qux, synth_E, 0

A simple example is combining real documents with synthetically generated documents for each query.

.. code-block:: python

    real_mqrel = MaterializedQRelConfig(
        ...
        corpus_path='real_corpus.jsonl',
        qrel_path='real_qrels.tsv',
    )
    synth_mqrel = MaterializedQRelConfig(
        ...
        corpus_path='llama_corpus.jsonl',
        qrel_path='llama_qrels.tsv',
    )

    dataset = MultiLevelDataset(qrel_config=[real_mqrel, synth_mqrel], ...)


The above snippet results in a dataset with these records:


.. code-block:: text

    ## format: (query, passage, label)

    ## combined data

    foo, real_A, 1
    foo, real_B, 0
    foo, synth_A, 3
    foo, synth_B, 1
    foo, synth_C, 0
    bar, real_C, 1
    bar, real_D, 0
    qux, synth_D, 3
    qux, synth_E, 0

But this is not good.
Because the range of labels is different in the combined dataset, the real positive documents (``real_A`` and ``real_C``) are used as irrelevant documents which is not correct.

In a more complex pipeline, we assign label ``3`` to real positives before merging them.
To make the pipeline more interesting, we filter the real negatives and only keep the real annotated positives.

.. code-block:: python

    real_mqrel = MaterializedQRelConfig(
        ...
        corpus_path='real_corpus.jsonl',
        qrel_path='real_qrels.tsv',
        # only choose positives
        min_score=1,
        # match relevancy level of positive synthetic documents
        score_transform=3
    )
    synth_mqrel = MaterializedQRelConfig(
        ...
        corpus_path='llama_corpus.jsonl',
        qrel_path='llama_qrels.tsv',
    )

    dataset = MultiLevelDataset(qrel_config=[real_mqrel, synth_mqrel], ...)

With the above snippet, we get a dataset with the following records (note that compared to the previous snippet, the label of ``real_A`` and ``real_C`` has changed to ``3`` and ``real_B`` and ``real_D`` are removed).

.. code-block:: text

    ## format: (query, passage, label)

    ## combined data

    foo, real_A, 3
    foo, synth_A, 3
    foo, synth_B, 1
    foo, synth_C, 0
    bar, real_C, 3
    qux, synth_D, 3
    qux, synth_E, 0


.. _custom-qrel-formats:

Custom File Formats
-----------------------------------


Trove allows you to read the annotations (qrel files) from custom file formats.
You just need to register a function that can read that file format.

.. code-block:: python

    @trove.register_loader('qrel')
    def load_qrel_from_custom_format(filepath, num_proc=None) -> datasets.Dataset:
        ...

.. tip::

    See all registered file loaders with ``trove.available_loaders()``.


Your loader function should first check if it can load the given file.
If it cannot load the file, it should return ``None``. This is the mechanism that Trove uses to find the function that can load each input file.
For example, a function that loads CSV files should return ``None`` if it receives a pickle file.
Similarly, if the schema in a given file does not match its expected schema, it should also return ``None`` (e.g., missing columns in a CSV file).

Each qrel loader function must return an instance of huggingface ``datasets.Dataset`` with ``'qid'``, ``'docid'``, and ``'score'`` columns.

    * ``'qid'`` is of type `str` and represents the query ID for the record.

    * ``'docid'`` is a list of string values (``List[str]``),
      where each item is the ID of one annotated document for this query.

    * ``'score'`` is a list of `int` or `float` values (``List[Union[int, float]]``). For the `i_th` record, ``loaded_qrel[i]['score'][idx]`` is the annotation for document ``loaded_qrel[i]['docid'][idx]``.

See :func:`~trove.data.file_reader.register_loader` documentation for more details.
Look at existing loaders like :func:`trove.data.file_reader_functions.qrel_from_csv` as examples.

Data Collator
---------------------

Trove offers a simple data collator that handles tokenization, padding, etc.
It also adds an ``eos`` token at the end of all sequences if necessary (this is useful when using last-token pooling).
You can pass an instance of this data collator to |RetrievalTrainer| or |RetrievalEvaluator|.

.. code-block:: python

    data_args = trove.DataArguments(query_max_len=128, ...)
    tokenizer = transformers.AutoTokenizer.from_pretrained(...)
    collator = trove.RetrievalCollator(data_args=data_args, tokenizer=tokenizer, append_eos=True)
    trainer = trove.RetrievalTrainer(data_collator=data_collator, ...)

Memory Consumption
---------------------

Trove uses memory mapped apache arrow tables (through huggingface datasets) to reduce memory consumption without noticeable impact on performance.
As a result, we can easily work with tens of millions of records without any issues.

Although not needed for most cases, cached datasets eliminate this small memory and performance overheads altogether while keeping Trove's data processing capabilities.
Cached datasets do not have any memory or performance overheads.
It is just a Huggingface ``datasets.Dataset`` instance reading records from a JSONL file.

Trove datasets provide two methods to cache the processed dataset to disk and offload the intermediate results.

* ``export_and_load_train_cache()`` saves all training records in a JSON lines files and returns a fresh instance of the same dataset that reads the training records from cache (both |BinaryDataset| and |MultiLevelDataset| provide this method).
* ``export_and_load_train_cache()`` saves several files with information needed for evaluation. It also returns a new instance of the same dataset that is based on the cached data (only |MultiLevelDataset| provides this method).

.. attention::

   ``export_and_load_eval_cache()`` does not have a significant overhead.
   But, depending on the size of your dataset, caching training records (with ``export_and_load_train_cache()``) might take long the very first time.
   Also based on the size of your dataset, caching training records might lead to very large files.
   Remember to delete the cache if you do not need it in the future.
   The cache is located in the huggingface hub cache directory (usually ``$HOME/.cache/huggingface/assets``).


You should call the appropriate ``export_and_load*`` method every time you want to use a cached dataset; but it does **not** create new cache files for every call.
If the data files and processing logic remain the same, datasets load existing cache files if available.


.. code-block:: python

    dataset = MultiLevelDataset(...)
    # For datasets using in train loop
    # This includes both training dataset and evaluation datasets used to approximate IR metrics during training
    dataset = dataset.export_and_load_train_cache()
    # for datasets used for exact evaluation of retrievers after training (often with trove.RetrievalEvaluator class)
    dataset = dataset.export_and_load_eval_cache()


BinaryDataset
------------------------------

|BinaryDataset| is a more limited class compared to |MultiLevelDataset|: it can only represent binary label values and cannot be used for any sort of evaluation.
But, it is helpful for contrastive training with InfoNCE loss.

Keep the following in mind when using |BinaryDataset|

* |BinaryDataset| ignores the label values. Instead, it maintains two collections of documents: positives and negatives.
  Regardless of the label values in qrel files, it always assigns labels 1 and 0 to positive and negative documents, respectively.

* To create a training instance for a given query, it samples one document from the positive collection and ``group_size-1`` documents from the negative collection.

* In each training instance (i.e., ``{'query': '...', 'passage': [...], 'label': [...]}``), the positive document is always the first element of the list (the left most element).

* As mentioned earlier, when using Trove's builtin InfoNCE loss (|InfoNCELoss|), you must use an instance of |BinaryDataset| for training. See :ref:`modeling-loss-functions` for more details.

* It uses the sampling technique used in `Tevatron Library <https://github.com/texttron/tevatron>`_ to select positive and negative documents for each query.
