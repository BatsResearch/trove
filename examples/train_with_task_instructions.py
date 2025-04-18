"""Training a dense retriever, optionally with task instructions."""

from typing import Dict, Optional

from transformers import AutoTokenizer, HfArgumentParser

from trove import (
    BiEncoderRetriever,
    BinaryDataset,
    DataArguments,
    MaterializedQRelConfig,
    ModelArguments,
    RetrievalCollator,
    RetrievalTrainer,
    RetrievalTrainingArguments,
)
from trove.modeling.encoder_default import DefaultEncoder

#####################################################################################
#####################################################################################
## Adding a custom Encoder with instructions.
## This part is not needed if you don't want to use task instructions.
#####################################################################################
#####################################################################################


def create_task_to_instruct_mapping() -> Dict[str, str]:
    # Instructions are from https://arxiv.org/pdf/2401.00368
    task_name_to_instruct: Dict[str, str] = {
        "ArguAna": "Given a claim, find documents that refute the claim",
        "ClimateFEVER": "Given a claim about climate change, retrieve documents that support or refute the claim",
        "DBPedia": "Given a query, retrieve relevant entity descriptions from DBPedia",
        "FEVER": "Given a claim, retrieve documents that support or refute the claim",
        "FiQA2018": "Given a financial question, retrieve user replies that best answer the question",
        "HotpotQA": "Given a multi-hop question, retrieve documents that can help answer the question",
        "MSMARCO": "Given a web search query, retrieve relevant passages that answer the query",
        "NFCorpus": "Given a question, retrieve relevant documents that best answer the question",
        "NQ": "Given a question, retrieve Wikipedia passages that answer the question",
        "QuoraRetrieval": "Given a question, retrieve questions that are semantically equivalent to the given question",
        "SCIDOCS": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
        "SciFact": "Given a scientific claim, retrieve documents that support or refute the claim",
        "Touche2020": "Given a question, retrieve detailed and persuasive arguments that answer the question",
        "TRECCOVID": "Given a query on COVID-19, retrieve documents that answer the query",
        # C-MTEB eval instructions
        "T2Retrieval": "Given a Chinese search query, retrieve web passages that answer the question",
        "MMarcoRetrieval": "Given a web search query, retrieve relevant passages that answer the query",
        "DuRetrieval": "Given a Chinese search query, retrieve web passages that answer the question",
        "CovidRetrieval": "Given a question on COVID-19, retrieve news articles that answer the question",
        "CmedqaRetrieval": "Given a Chinese community medical question, retrieve replies that best answer the question",
        "EcomRetrieval": "Given a user query from an e-commerce website, retrieve description sentences of relevant products",
        "MedicalRetrieval": "Given a medical question, retrieve user replies that best answer the question",
        "VideoRetrieval": "Given a video search query, retrieve the titles of relevant videos",
    }

    # add lower case keys to match some beir names
    task_name_to_instruct.update(
        {k.lower(): v for k, v in task_name_to_instruct.items()}
    )
    # other cases where lower case match still doesn't work
    task_name_to_instruct["trec-covid"] = task_name_to_instruct["TRECCOVID"]
    task_name_to_instruct["climate-fever"] = task_name_to_instruct["ClimateFEVER"]
    task_name_to_instruct["dbpedia-entity"] = task_name_to_instruct["DBPedia"]
    task_name_to_instruct["webis-touche2020"] = task_name_to_instruct["Touche2020"]
    task_name_to_instruct["fiqa"] = task_name_to_instruct["FiQA2018"]
    task_name_to_instruct["quora"] = task_name_to_instruct["QuoraRetrieval"]

    # for miracl evaluation
    task_name_to_instruct["miracl"] = (
        "Given a question, retrieve Wikipedia passages that answer the question"
    )
    task_name_to_instruct["cqadupstack"] = (
        "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question"
    )
    return task_name_to_instruct


class EncoderWithTaskInstructions(DefaultEncoder):
    _alias = "encoder_with_task_instructs"

    def __init__(self, *args, **kwargs) -> None:
        """Encoder with task instructions.

        We inherit from ``DefaultEncoder`` which already handles loading/saving/encoding/etc. If
        you want to have full control, then inherit from ``trove.PretrainedEncoder`` instead.
        """

        super().__init__(*args, **kwargs)

        self.task2instruct_mapping = create_task_to_instruct_mapping()

    def get_task_instruction(self, task_name: str) -> str:
        task_name = task_name.lower().replace("_", "-")
        if task_name.startswith("cqadupstack"):
            task_name = "cqadupstack"
        ins = self.task2instruct_mapping[task_name]
        ins = ins.strip()
        return ins

    def format_query(self, text: str, dataset: str, **kwargs) -> str:
        task_description = self.get_task_instruction(task_name=dataset)
        query_fmt = f"Instruct: {task_description}\nQuery: {text.strip()}"
        return query_fmt

    def format_passage(self, text: str, title: Optional[str] = None, **kwargs) -> str:
        if title is not None:
            text = f"{title.strip()} {text.strip()}"
        text = text.strip()
        return text


#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################


def main():
    # parse arguments
    parser = HfArgumentParser(
        (RetrievalTrainingArguments, ModelArguments, DataArguments)
    )
    train_args, model_args, data_args = parser.parse_args_into_dataclasses()

    ##################################
    ## create the model and tokenizer
    ##################################
    with train_args.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = BiEncoderRetriever.from_model_args(
        args=model_args, training_args=train_args
    )

    #####################################
    ## create dataset and data collator
    #####################################
    data_args.dataset_name = "msmarco"

    files = dict(
        qrel_path="hf://datasets/BatsResearch/trove-examples-data/tevatron_msmarco_passage_aug_qrel.jsonl",
        corpus_path="hf://datasets/mteb/msmarco@c5a29a104738b98a9e76336939199e264163d4a0/corpus.jsonl",
        query_path="hf://datasets/mteb/msmarco@c5a29a104738b98a9e76336939199e264163d4a0/queries.jsonl",
    )
    with train_args.main_process_first():
        pos_conf = MaterializedQRelConfig(min_score=1, **files)
        neg_conf = MaterializedQRelConfig(max_score=1, **files)
        dataset = BinaryDataset(
            data_args=data_args,
            positive_configs=[pos_conf],
            negative_configs=[neg_conf],
            format_query=model.format_query,
            format_passage=model.format_passage,
            num_proc=8,
        )
    data_collator = RetrievalCollator(
        data_args=data_args,
        tokenizer=tokenizer,
        append_eos=model.append_eos_token,
    )
    #########################################
    ## train the model
    #########################################
    trainer = RetrievalTrainer(
        args=train_args,
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
