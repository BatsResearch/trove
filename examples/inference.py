"""Evaluation and hard negative mining with dense retrievers."""

from dataclasses import dataclass

from transformers import AutoTokenizer, HfArgumentParser

from trove import (
    BiEncoderRetriever,
    DataArguments,
    EvaluationArguments,
    MaterializedQRelConfig,
    ModelArguments,
    MultiLevelDataset,
    RetrievalCollator,
    RetrievalEvaluator,
)


@dataclass
class ScriptArguments:
    job: str
    """Choose one of 'evaluate' or 'mine_hn'."""

    def __post_init__(self):
        assert self.job in ["evaluate", "mine_hn"]


def main():
    # parse arguments
    parser = HfArgumentParser(
        (EvaluationArguments, ModelArguments, DataArguments, ScriptArguments)
    )
    eval_args, model_args, data_args, script_args = parser.parse_args_into_dataclasses()

    ##################################
    ## create the model and tokenizer
    ##################################
    with eval_args.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = BiEncoderRetriever.from_model_args(args=model_args)

    #####################################
    ## create dataset and data collator
    #####################################
    data_args.dataset_name = "scifact"

    if script_args.job == "evaluate":
        qrel_path = "hf://datasets/mteb/scifact@d56462d0e63a25450459c4f213e49ffdb866f7f9/qrels/test.tsv"
    else:
        qrel_path = "hf://datasets/mteb/scifact@d56462d0e63a25450459c4f213e49ffdb866f7f9/qrels/train.tsv"

    with eval_args.main_process_first():
        mqrel_conf = MaterializedQRelConfig(
            qrel_path=qrel_path,
            corpus_path="hf://datasets/mteb/scifact@d56462d0e63a25450459c4f213e49ffdb866f7f9/corpus.jsonl",
            query_path="hf://datasets/mteb/scifact@d56462d0e63a25450459c4f213e49ffdb866f7f9/queries.jsonl",
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

    #########################################
    ## Create Evaluator and run the task
    #########################################
    evaluator = RetrievalEvaluator(
        args=eval_args,
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        eval_dataset=dataset,
    )
    if script_args.job == "evaluate":
        evaluator.evaluate()
    else:
        evaluator.mine_hard_negatives()


if __name__ == "__main__":
    main()
