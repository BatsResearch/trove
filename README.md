# Trove

_A Flexible Toolkit for Dense Retrieval_

<br>

<p align="center"><img width=300 alt="Trove Logo" src="https://huggingface.co/datasets/BatsResearch/trove-lib-documentation-assets/resolve/main/logo/logo_no_background.svg"/></p>

______________________________________________________________________

Trove is a lightweight toolkit for training and evaluating transformer-based dense retrievers.
It aims to keep the codebase simple and hackable, while offering a clean, unified interface for quickly experimenting with new ideas.

**Key features**:

- Well-documented and easy-to-understand codebase
- Simple, modular design that's easy to extend and integrate into different workflows
- Minimal, consistent interface for evaluation and hard negative mining
- Built to work seamlessly with the Hugging Face ecosystem (e.g., PEFT methods, distributed training/inference)
- Effortless manipulation and combination of multiple datasets on-the-fly

[🎓 **Documentation**](https://batsresearch.github.io/trove/)

[📚 **Examples**](https://github.com/BatsResearch/trove/tree/main/examples)

⭐ Check out our recent [paper](https://www.arxiv.org/abs/2503.23239) (and [code](https://github.com/BatsResearch/sycl)) to see how Trove's data manipulation capabilities enable us to train retrievers with synthetic multi-level ranking contexts.

## Quick Tour

Install Trove from PyPI:

```bash
pip install ir-trove
```

To get the latest changes, install from source:

```bash
pip install git+https://github.com/BatsResearch/trove
```

Import the necessary modules for training and evaluation:

```python
from transformers import AutoTokenizer, HfArgumentParser
from trove import (
    BiEncoderRetriever,
    BinaryDataset,
    DataArguments,
    EvaluationArguments,
    MaterializedQRelConfig,
    ModelArguments,
    MultiLevelDataset,
    RetrievalCollator,
    RetrievalEvaluator,
    RetrievalTrainer,
    RetrievalTrainingArguments,
)
```


### Training

> [Documentation](https://batsresearch.github.io/trove/guides/training.html)

Train with binary labels:

```python
parser = HfArgumentParser((RetrievalTrainingArguments, ModelArguments, DataArguments))
train_args, model_args, data_args = parser.parse_args_into_dataclasses()

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
model = BiEncoderRetriever.from_model_args(args=model_args)

pos = MaterializedQRelConfig(min_score=1, qrel_path="train_qrel.tsv", corpus_path="corpus.jsonl", query_path="queries.jsonl")
neg = MaterializedQRelConfig(max_score=1, qrel_path="train_qrel.tsv", corpus_path="corpus.jsonl", query_path="queries.jsonl")
dataset = BinaryDataset(data_args=data_args, positive_configs=pos, negative_configs=neg, format_query=model.format_query, format_passage=model.format_passage)
data_collator = RetrievalCollator(data_args=data_args, tokenizer=tokenizer, append_eos=model.append_eos_token)

trainer = RetrievalTrainer(args=train_args, model=model, tokenizer=tokenizer, data_collator=data_collator, train_dataset=dataset)
trainer.train()
```

To train with graduated relevance labels (e.g., `{0, 1, 2, 3}`), you just need to change a few lines:

```python
...
conf = MaterializedQRelConfig(qrel_path="train_qrel.tsv", corpus_path="corpus.jsonl", query_path="queries.jsonl")
dataset = MultiLevelDataset(data_args=data_args, qrel_config=conf, format_query=model.format_query, format_passage=model.format_passage)
...
```

### Data Manipulation

> [Documentation](https://batsresearch.github.io/trove/guides/data.html)

Manipulate and combine multiple data sources with just a few lines of code.
The following snippet combines a multi-level synthetic dataset (with labels `{0, 1, 2, 3}`) with real annotated positives and two mined hard negatives per query.
Before merging, it also reassigns the label values: real positives are labeled `3`, and mined negatives are labeled `1`.

```python
...
real_pos = MaterializedQRelConfig(min_score=1, score_transform=3, corpus_path="real_corpus.jsonl", qrel_path="qrels/train.tsv", query_path="queries.jsonl")
mined_neg = MaterializedQRelConfig(group_random_k=2, score_transform=1, corpus_path="real_corpus.jsonl", qrel_path="mined_qrel.tsv", query_path="queries.jsonl")
synth_data = MaterializedQRelConfig(corpus_path="corpus_multilevel_synth.jsonl", qrel_path="qrel_multilevel_synth.tsv", query_path="queries.jsonl")

dataset = MultiLevelDataset(qrel_config=[real_pos, mined_neg, synth_data], data_args=data_args, format_query=model.format_query, format_passage=model.format_passage)
...
```

### Inference

> [Documentation](https://batsresearch.github.io/trove/guides/inference.html)

**Evaluation:** Calculate IR metrics

```python
parser = HfArgumentParser((EvaluationArguments, ModelArguments, DataArguments))
eval_args, model_args, data_args = parser.parse_args_into_dataclasses()

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
model = BiEncoderRetriever.from_model_args(args=model_args)

conf = MaterializedQRelConfig(qrel_path="test_qrel.tsv", corpus_path="corpus.jsonl", query_path="queries.jsonl")
dataset = MultiLevelDataset(data_args=data_args, qrel_config=conf, format_query=model.format_query, format_passage=model.format_passage)
data_collator = RetrievalCollator(data_args=data_args, tokenizer=tokenizer, append_eos=model.append_eos_token)

evaluator = RetrievalEvaluator(args=eval_args, model=model, tokenizer=tokenizer, data_collator=data_collator, eval_dataset=dataset)
evaluator.evaluate()
```

**Hard Negative Mining:** With very minor changes, you can use the above snippet to mine hard negatives for the given queries.
You only need to change the last line:

```python
...
evaluator.mine_hard_negatives()
...
```

### Distributed Environments

Trove supports **both training and inference** in multi-gpu and multi-node environments.
You just need to run your scripts with a distributed launcher.

```bash
accelerate launch --multi_gpu {train.py or eval.py} {script arguments}
```

You can also use deepspeed for training.
Since Trove wraps around and is fully compatible with huggingface transformers, you just need to pass your deepspeed config file as a command line argument.
See [huggingface transformers](https://huggingface.co/docs/transformers/en/deepspeed) documentation for more details.

## Citation

If you use this software, please cite us.

```bibtex
@misc{esfandiarpoortrove,
  author = {Reza Esfandiarpoor and Stephen H. Bach},
  title = {Trove: A Flexible Toolkit for Dense Retrieval},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/BatsResearch/trove}}
}
```

## Acknowledgment

Some of the high-level design choices are inspired by [Tevatron](https://github.com/texttron/tevatron) library.
Trove also adapts some implementation details of [Tevatron](https://github.com/texttron/tevatron).
Some data manipulations are inspired by ideas in [Huggingface datasets](https://github.com/huggingface/datasets) source code.

This material is based upon work supported by the National Science Foundation under Grant No. RISE-2425380. Any
opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and
do not necessarily reflect the views of the National Science Foundation. Disclosure: Stephen Bach is an advisor
to Snorkel AI, a company that provides software and services for data-centric artificial intelligence.
