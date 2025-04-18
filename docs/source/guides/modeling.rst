Modeling
==================

.. role:: raw-html(raw)
    :format: html


.. |BinaryDataset| replace:: :py:class:`~trove.data.ir_dataset_binary.BinaryDataset`
.. |PretrainedEncoder| replace:: :py:class:`~trove.modeling.pretrained_encoder.PretrainedEncoder`
.. |DefaultEncoder| replace:: :py:class:`~trove.modeling.encoder_default.DefaultEncoder`
.. |PretrainedRetriever| replace:: :py:class:`~trove.modeling.pretrained_retriever.PretrainedRetriever`
.. |BiEncoderRetriever| replace:: :py:class:`~trove.modeling.retriever_biencoder.BiEncoderRetriever`
.. |ModelArguments| replace:: :py:class:`~trove.modeling.model_args.ModelArguments`
.. |RetrievalLoss| replace:: :py:class:`~trove.modeling.loss_base.RetrievalLoss`
.. |InfoNCELoss| replace:: :py:class:`~trove.modeling.losses.InfoNCELoss`



Our goal is to be compatible with existing huggingface transformers ecosystem (e.g., PEFT and distributed training) and maintain this compatibility in the future with minimal changes to Trove.
Trove's goal is not to support and cover everything right out of the box.
Instead, we want to keep the code simple and flexible so users can easily adapt it for their use case.

To achieve this, Trove models rely on an ``encoder`` object that encapsulates the most dynamic aspects of
modeling like supporting different PEFT techniques or implementing new retrieval methods (e.g., retrieval using task instructions).
We provide an optional abstraction and some helper functions to help with creating the encoder object.
For maximum flexibility, Trove also accepts any arbitrary encoder object provided by the user, with minimal limitations to remain compatible with huggingface transformers.


Trove Models
---------------------

Retriever variants (e.g., |BiEncoderRetriever|) are the main classes in Trove, i.e., retriever is what we use as `model` in training/inference scripts.
Each retriever has an ``encoder`` attribute that is responsible for everything related to the backbone transformers model (e.g., Contriever).
For example, ``encoder`` object should save/load the checkpoints and provide the logic for calculating the embedding vectors (e.g., pooling, normalization).

Trove provides three options for using a transformers model as encoder.

* :ref:`arbitrary-torch-encoder`
* :ref:`default-encoder-wrapper`
* :ref:`custom-encoder-wrapper`


.. _arbitrary-torch-encoder:

Arbitrary torch Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can create a retriever with an instance of ``torch.nn.Module`` as encoder, as long as it provides certain methods (see below).

.. code-block:: python

    my_custom_encoder: torch.nn.Module = ...
    args = trove.ModelArguments(loss='infonce')
    model = trove.BiEncoderRetriever(encoder=my_custom_encoder, model_args=args)

.. warning::

    For training, you must make sure your custom encoder is compatible with huggingface ``transformers.Trainer``.


Trove expects each encoder to provide several methods

    * ``encode_query(inputs) -> torch.Tensor``: function that takes batched query tokens as inputs and returns the embedding vectors.
    * ``encode_passage(inputs) -> torch.Tensor``: function that takes batched passage tokens as inputs and returns the embedding vectors.
    * ``save_pretrained()``: the encoder must provide this method if we need to save checkpoints. The signature is the same as that of huggingface transformers models.
    * ``similarity_fn(query: torch.Tensor, passage: torch.Tensor) -> torch.Tensor`` It is optional. The default is the dot product between query and passage embeddings.

.. warning::

    Trove retrievers like |BiEncoderRetriever| provide other methods like ``format_query`` and ``format_passage`` and attributes like ``append_eos_token``.
    These are valid only if your encoder provides methods and attributes with the same names or if you pass these as arguments to retrievers `__init__` method.
    Otherwise, these are set to default values which might not be valid for your specific encoder.

.. _default-encoder-wrapper:

Trove Default Wrapper (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Trove provides |DefaultEncoder| as a general encoder wrapper that can load and save huggingface transformers models.
It supports quantization and LORA adapters which you can specify through |ModelArguments|.
You also need to choose a pooling method (options are ``first_token``, ``last_token``, and ``mean``) and specify whether to normalize the embeddings or not.

You do not need to instantiate |DefaultEncoder| manually.
You just provide the model arguments to the retriever and it instantiates the encoder for you.

.. code-block:: python

    args = trove.ModelArguments(
        model_name_or_path="mistralai/Mistral-7B-v0.1",
        encoder_class="default",
        pooling="last_token",
        normalize="yes",
        use_peft=True,
        loss='infonce'
    )
    model = trove.BiEncoderRetriever.from_model_args(args=model_args)


.. _custom-encoder-wrapper:

Trove Custom Wrappers (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


For maximum flexibility, you can create your own custom encoder wrappers.
For example to include formatting functions to add task instructions to queries.

Trove encoder wrappers are subclasses of |PretrainedEncoder|.
Subclasses of |PretrainedEncoder| should take care of everything that is needed to use (train or evaluate) an encoder.
This includes but is not limited to:

* loading the model
* loading/merging LORA adapters (optional)
* quantization (optional)
* how to format queries and passages (e.g., use task instructions)
* how to calculate embeddings (e.g., pooling type)
* etc.

.. note::

    Some requirements like providing query and passage formatting methods are not necessary for the function of the encoder class itself.
    But the goal is to keep all the details related to each backbone in one place instead of handling them in user scripts.
    For instance, instead of selecting the correct instructions for each model, we prefer to just instantiate the correct encoder wrapper for that class
    and let it handle the rest.

Here, we use an example to explain the step-by-step process for creating an encoder class that loads a transformers model as the backbone.
It supports using quantization and LORA adapters for training.
We use last-token pooling and normalize the embedding vectors.

Name and Alias
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, we select an alias for our custom encoder class.

.. code-block:: python

   class MyEncoder(trove.PretrainedEncoder):
        _alias = 'simple_encoder_wrapper'
        ...


Now to use this class, we should set the value of ``model_args.encoder_class`` to either ``MyEncoder`` or ``simple_encoder_wrapper`` in our final script.
There are examples later in this guide.


Loading (with LORA and Quantization)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We load the model in the `__init__` method.

.. code-block:: python

    import peft
    from transformers import BitsAndBytesConfig, AutoModel

    class MyEncoder(trove.PretrainedEncoder):
        ...
        def __init__(self, args: trove.ModelArguments, **kwargs):
            q_config = BitsAndBytesConfig(bnb_4bit_quant_type=args.bnb_4bit_quant_type, ...)
            model = AutoModel.from_pretrained(args.model_name_or_path, quantization_config=q_config)
            model = peft.prepare_model_for_kbit_training(model)
            lora_conf = peft.LoraConfig(r=args.lora_r, ...)
            self.model = peft.get_peft_model(model, lora_config)

.. note::

    The above example always quantizes the parameters and adds LORA adapters.
    You can use |ModelArguments| attributes (e.g., ``use_peft``) to dynamically decide whether to use quantization or LORA adapters and select the configuration.
    If more config options are needed, you can subclass |ModelArguments| and add new attributes.

Next, to take full advantage of ``transformers.Trainer`` module, we expose some methods and attributes of the backbone model.

.. code-block:: python

    from trove.modeling import modeling_utils

    class MyEncoder(trove.PretrainedEncoder):
        ...
        def __init__(self, args: trove.ModelArguments, **kwargs):
            ...
            modeling_utils.add_model_apis_to_wrapper(wrapper=self, model=self.model)

Finally, since we want to use last-token pooling, we set ``append_eos_token`` to ``True``, which signals that this class expects an ``eos`` token at the end of all input sequences.
This is not used by the encoder class itself.
But, user scripts can rely on it to automatically configure data processing attributes for each encoder.

.. code-block:: python

    class MyEncoder(trove.PretrainedEncoder):
        ...
        def __init__(self, args: trove.ModelArguments, **kwargs):
            ...
            self.append_eos_token = True

Saving
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We add a ``save_pretrained`` method to save the checkpoints.
To be able to load the model in other frameworks, we only save the checkpoint for the main huggingface model (independent of Trove).

.. code-block:: python

    def save_pretrained(self, *args, **kwargs):
        if "state_dict" in kwargs:
            kwargs["state_dict"] = {
                k.removeprefix('model.'): v for k, v in kwargs["state_dict"].items()
            }
        return self.model.save_pretrained(*args, **kwargs)


Input Formatting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As preprocessing, we prefix inputs with ``query:`` and ``passage``::

    def format_query(self, text: str, **kwargs) -> str:
        return "Query: " + text.strip()

    def format_passage(self, text: str, **kwargs) -> str:
        return "Passage: " + text.strip()

You can also implement more complex strategies.

.. tip::

    When calling ``format_query`` and ``format_passage``, Trove datasets also pass the name of the dataset as an extra keyword argument named ``dataset``.
    It allows us to customize the formatting for each dataset (e.g., use different task instructions for each dataset).


Calculating the Embeddings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use last-token pooling and normalize the embeddings.

.. code-block:: python

    def encode(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        reps = last_hidden_state[
            torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths
        ]
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    def encode_query(self, inputs):
        return self.encode(inputs=inputs)

    def encode_passage(self, inputs):
        return self.encode(inputs=inputs)

This is all we need to do to add a custom encoder wrapper to Trove.

Use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are two ways to use our encoder wrapper in training or evaluation scripts.

**Manually**

We can instantiate the encoder manually::

    args = trove.ModelArguments(model_name_or_path='facebook/contriever', loss='infonce')
    encoder = MyEncoder(args)
    model = trove.BiEncoderRetriever(encoder=encoder, model_args=args)

**Name or Alias**

We can use the wrapper name or alias and let the retriever instantiate the encoder::

    # you can also set `encoder_class="MyEncoder"`
    args = trove.ModelArguments(encoder_class='simple_encoder_wrapper', model_name_or_path='facebook/contriever', loss='infonce')
    model = trove.BiEncoderRetriever.from_model_args(args)

.. _modeling-loss-functions:

Loss Functions
---------------------

If a loss function is already registered with Trove, you just need to specify its name in your model arguments.
The retriever class automatically instantiates the corresponding loss class.

.. tip::

    Use ``trove.RetrievalLoss.available_losses()`` to see the name of all available loss functions.


For example, to use ``infonce``, you can do this::

    args = trove.ModelArguments(loss='infonce', ...)
    model = trove.BiEncoderRetriever.from_model_args(args=args)
    # how you instantiate your encoder does not impact the loss function
    # this instantiates the same loss class
    encoder = MyEncoder(args)
    model = trove.BiEncoderRetriever(encoder=encoder, model_args=args)


.. tip::

    If a loss function supports or expects extra keyword arguments in its `__init__` method, you can pass those keyword arguments
    by ``loss_extra_kwargs`` argument of the retriever like ``trove.BiEncoderRetriever.from_model_args(args=args, loss_extra_kwargs={...})``

.. attention::

    When using Trove's builtin InfoNCE loss (|InfoNCELoss|), you must use an instance of |BinaryDataset| for training.
    |InfoNCELoss| ignores the given labels. Instead, it assumes the positive is the very first item in the list of passages for each query.


Custom Loss Functions
~~~~~~~~~~~~~~~~~~~~~~

You can easily implement and register a new loss function with Trove.
You just need to create a subclass of |RetrievalLoss| that implements your loss function.

Let's go through an example that implements the KL divergence loss.
Note that KL loss is already implemented in Trove and you can use it by setting ``model_args.loss="kl"``.

First, we inherit from |RetrievalLoss| and parse the arguments.

.. code-block:: python

    class MyCustomKLLoss(RetrievalLoss):
        _alias = "custom_kl"

        def __init__(self, args: ModelArguments, **kwargs) -> None:
            super().__init__()
            self.temperature = args.temperature

Next, we implement the ``forward`` method that calculates the loss value.

.. code-block:: python

    def forward(self, logits: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        if label.size(1) != logits.size(1):
            label = torch.block_diag(*torch.chunk(label, label.shape[0]))

        preds = F.log_softmax(logits / self.temperature, dim=1)
        targets = F.log_softmax(label.double(), dim=1)
        loss = F.kl_div(
            input=preds, target=targets, log_target=True, reduction="batchmean"
        )
        return loss


``logits`` are similarity scores between all queries and all documents.
In a distributed environment with multiple processes, ``logits`` includes the similarity scores even for in-batch negatives.
But, ``label`` only has enteries for labeled documents, and not for in-batch negatives (e.g., only for positives and mined hard negatives).
So, shape of ``label`` and ``logits`` diverges.
To make label and logits sizes match, we assign label zero (0) to all in-batch negatives and expand ``label`` matrix by::

    label = torch.block_diag(*torch.chunk(label, label.shape[0]))

To use the new loss function, we just need to specify its name in model arguments::

    model_args = ModelArguments(loss="custom_kl", ...)
    # or
    model_args = ModelArguments(loss="MyCustomKLLoss", ...)


Retrieval Logic
---------------------

As you have seen so far, the retriever class is the main `model` class used in training and evaluation scripts.
Trove implements the bi-encoder retrieval logic (|BiEncoderRetriever|), which encodes the query and document separately and then calculates their similarity based on some metric like dot product.

Here is an example that shows how to use the retriever class.
See |PretrainedRetriever| and |BiEncoderRetriever| documentation for more details.

.. code-block:: python

    model = trove.BiEncoderRetriever.from_model_args(...)
    # embed queries
    query_embs = model.encode_query(query_tokens)
    # embed passages
    passage_embs = model.encode_passage(passage_tokens)
    # full forward pass
    output = model(query=query_tokens, passage=passage_tokens, label=labels) # label is optional
    print(output.query.shape) # query embeddings
    print(output.passage.shape) # passage embeddings
    print(output.logits.shape) # query-passage similarity scores
    # if lables are given and retriever is instantiated with a loss module
    print(output.loss)

Custom Retrieval Logic
~~~~~~~~~~~~~~~~~~~~~~~~~~

To implement a new retrieval logic, you need to create a subclass of |PretrainedRetriever| and implement the ``forward()`` method.
See |PretrainedRetriever| documentation for signature of the ``forward()`` method.
You can follow |BiEncoderRetriever| code as an example.
