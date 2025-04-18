"""Training a dense retriever."""

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
