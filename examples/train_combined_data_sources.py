"""Training a dense retriever on a combination of real and synthetic data."""

from transformers import AutoTokenizer, HfArgumentParser

from trove import (
    BiEncoderRetriever,
    DataArguments,
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

    #####################################
    ## create dataset and data collator
    #####################################
    data_args.dataset_name = "msmarco"

    real_corpus = "hf://datasets/mteb/msmarco@c5a29a104738b98a9e76336939199e264163d4a0/corpus.jsonl"
    real_query = "hf://datasets/mteb/msmarco@c5a29a104738b98a9e76336939199e264163d4a0/queries.jsonl"
    real_orig_qrel = "hf://datasets/mteb/msmarco@c5a29a104738b98a9e76336939199e264163d4a0/qrels/train.tsv"
    real_mined_qrel = (
        "hf://datasets/BatsResearch/sycl/real_data/bm25_top_docs/train_qrels.jsonl"
    )
    synth_corpus = "hf://datasets/BatsResearch/sycl/llama33_70b/corpus.jsonl"
    synth_qrel = "hf://datasets/BatsResearch/sycl/llama33_70b/qrels/train.tsv"

    with train_args.main_process_first():
        real_pos = MaterializedQRelConfig(
            corpus_path=real_corpus,
            query_path=real_query,
            qrel_path=real_orig_qrel,
            score_transform=3,
            min_score=1,
        )
        real_mined_hn = MaterializedQRelConfig(
            corpus_path=real_corpus,
            query_path=real_query,
            qrel_path=real_mined_qrel,
            score_transform=1,
            group_random_k=2,
        )
        synth_data = MaterializedQRelConfig(
            query_path=real_query,
            corpus_path=synth_corpus,
            qrel_path=synth_qrel,
        )

    with train_args.main_process_first():
        dataset = MultiLevelDataset(
            data_args=data_args,
            qrel_config=[real_pos, real_mined_hn, synth_data],
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
