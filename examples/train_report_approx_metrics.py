"""Report approximate IR metrics during training."""

from transformers import AutoTokenizer, HfArgumentParser

from trove import (
    BiEncoderRetriever,
    BinaryDataset,
    DataArguments,
    IRMetrics,
    MaterializedQRelConfig,
    ModelArguments,
    MultiLevelDataset,
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

    ###########################################
    ## create train dataset and data collator
    ###########################################
    data_args.dataset_name = "msmarco"

    corpus_path = "hf://datasets/mteb/msmarco@c5a29a104738b98a9e76336939199e264163d4a0/corpus.jsonl"
    query_path = "hf://datasets/mteb/msmarco@c5a29a104738b98a9e76336939199e264163d4a0/queries.jsonl"
    qrel_train_with_mined_HN = "hf://datasets/BatsResearch/trove-examples-data/tevatron_msmarco_passage_aug_qrel.jsonl"

    with train_args.main_process_first():
        pos_conf = MaterializedQRelConfig(
            qrel_path=qrel_train_with_mined_HN,
            corpus_path=corpus_path,
            query_path=query_path,
            min_score=1,
        )
        neg_conf = MaterializedQRelConfig(
            qrel_path=qrel_train_with_mined_HN,
            corpus_path=corpus_path,
            query_path=query_path,
            max_score=1,
        )
    with train_args.main_process_first():
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

    ###########################################################################
    ## Create eval datasets to report approximate IR metrics during training
    ###########################################################################
    with train_args.main_process_first():
        qrel_trec20 = "hf://datasets/BatsResearch/sycl/real_data/trec20/qrel.tsv"
        queries_trec20 = (
            "hf://datasets/BatsResearch/sycl/real_data/trec20/queries.jsonl"
        )
        eval_mqrel = MaterializedQRelConfig(
            corpus_path=corpus_path,
            query_path=queries_trec20,
            qrel_path=qrel_trec20,
        )
    with train_args.main_process_first():
        arg_overrides = {
            "passage_selection_strategy": "most_relevant",
            "group_size": 100,
        }
        eval_dataset = MultiLevelDataset(
            data_args=data_args,
            format_query=model.format_query,
            format_passage=model.format_passage,
            qrel_config=eval_mqrel,
            data_args_overrides=arg_overrides,
            num_proc=8,
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
        eval_dataset=eval_dataset,
        compute_metrics=IRMetrics(k_values=[1, 10, 100]),
    )
    trainer.train(ignore_keys_for_eval=["query", "passage"])
    trainer.save_model()


if __name__ == "__main__":
    main()
