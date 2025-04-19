# Examples

This directory contains simple example scripts for training, evaluation, and hard negative mining with Trove.

First, install Trove with `pip`:

```bash
pip install ir-trove
```

To get the latest changes, install from source:

```bash
pip install git+https://github.com/BatsResearch/trove
```

There are other features and command line options that are not used in these examples.
Read the [documentation](https://batsresearch.github.io/trove/) for all options and features.

## Training

`train_simple.py` is a very basic example of training with Trove.

Use the following command to train on one machine with 2 GPUs using deepspeed:

```bash
deepspeed --include localhost:0,1 train_simple.py \
    --deepspeed 'deepspeed_config.json' \
    --output_dir './model_output/mycontriever' \
    --model_name_or_path 'facebook/contriever' \
    --encoder_class 'default' \
    --pooling 'mean' \
    --normalize 'no' \
    --loss 'infonce' \
    --temperature '1.0' \
    --trust_remote_code 'true' \
    --group_size 16 \
    --query_max_len 32 \
    --passage_max_len 128 \
    --report_to 'wandb' \
    --save_strategy 'epoch' \
    --per_device_train_batch_size 16 \
    --learning_rate '1e-5' \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --gradient_accumulation_steps 4 \
    --warmup_ratio '0.05' \
    --dataloader_num_workers 2 \
    --save_only_model true \
    --trove_logging_mode 'local_main'
```

To train with Lora adapters, just add `--use_peft` to the command line arguments.

If available, you can train on multiple nodes using the same script without any change.

**Report IR Metrics During Training**

`train_report_approx_metrics.py` is the same as previous script but it also calculates and reports an approximation of IR metrics during training.

```bash
deepspeed --include localhost:0,1 train_report_approx_metrics.py \
    --deepspeed 'deepspeed_config.json' \
    --output_dir './model_output/mycontriever' \
    --model_name_or_path 'facebook/contriever' \
    --encoder_class 'default' \
    --pooling 'mean' \
    --normalize 'no' \
    --loss 'infonce' \
    --temperature '1.0' \
    --trust_remote_code 'true' \
    --group_size 16 \
    --query_max_len 32 \
    --passage_max_len 128 \
    --report_to 'wandb' \
    --save_strategy 'epoch' \
    --per_device_train_batch_size 16 \
    --learning_rate '1e-5' \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --gradient_accumulation_steps 4 \
    --warmup_ratio '0.05' \
    --dataloader_num_workers 2 \
    --save_only_model true \
    --eval_strategy steps \
    --eval_steps 1000 \
    --per_device_eval_batch_size 16 \
    --eval_on_start 'false' \
    --batch_eval_metrics 'true' \
    --label_names 'label' \
    --trove_logging_mode 'local_main'
```

**Train with Task Instructions**

`train_with_task_instructions.py` adds a new encoder wrapper class that adds task instructions to the queries.
You can run it similar to other training examples. But, set the `--encoder_class` to `encoder_with_task_instructs` to use the new encoder.

For example, to train a mistral model with task instructions and LORA (similar to [e5-mistral-7b-instruct](https://arxiv.org/pdf/2401.00368)), you can run the following:

```bash
deepspeed --include localhost:0,1 train_with_task_instructions.py \
    --deepspeed 'deepspeed_config.json' \
    --output_dir './model_output/mymistral' \
    --model_name_or_path 'mistralai/Mistral-7B-v0.1' \
    --encoder_class 'encoder_with_task_instructs' \
    --pooling 'last_token' \
    --normalize 'yes' \
    --use_peft 'true' \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
    --loss 'infonce' \
    --temperature '1.0' \
    --trust_remote_code 'true' \
    --group_size 16 \
    --query_max_len 32 \
    --passage_max_len 128 \
    --report_to 'wandb' \
    --save_strategy 'epoch' \
    --per_device_train_batch_size 16 \
    --learning_rate '1e-5' \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing 'true' \
    --warmup_ratio '0.05' \
    --dataloader_num_workers 2 \
    --save_only_model true \
    --trove_logging_mode 'local_main'
```

**Modularity**

We do not need to create two separate scripts to train with and without task instructions (`train_simple.py` and `train_with_task_instructions.py`).
We can use command line arguments to choose the appropriate encoder class.
If we just run `train_with_task_instructions.py` with `--encoder_class=default` it behaves exactly the same as `train_simple.py`. So we can use the same script for both training setups.

**Train on Combined Data Sources**

`train_combined_data_sources.py` combines three data sources and trains the model with KL divergence loss.

For each query: it 1) selects all synthetic passages with graduated labels (i.e., possible labels are `{0, 1, 2, 3}`), 2) selects the real annotated positives and assigns them label `3` before merging, 3) selects two BM25 mined hard negatives and assigns them label `1` before merging.

```bash
deepspeed --include localhost:0,1 train_combined_data_sources.py \
    --deepspeed 'deepspeed_config.json' \
    --output_dir './model_output/mycontriever' \
    --model_name_or_path 'facebook/contriever' \
    --encoder_class 'default' \
    --pooling 'mean' \
    --normalize 'no' \
    --loss 'kl' \
    --temperature '1.0' \
    --trust_remote_code 'true' \
    --group_size 7 \
    --passage_selection_strategy='random' \
    --query_max_len 32 \
    --passage_max_len 128 \
    --report_to 'wandb' \
    --save_strategy 'epoch' \
    --per_device_train_batch_size 16 \
    --learning_rate '1e-5' \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --gradient_accumulation_steps 4 \
    --warmup_ratio '0.05' \
    --dataloader_num_workers 2 \
    --save_only_model true \
    --trove_logging_mode 'local_main'
```

## Evaluation

`inference.py` evaluates a model and reports the IR metrics.

The following command evaluates the model and then prints the IR metrics to terminal, log them to wandb, and saves them on disk.

```bash
python inference.py \
    --job 'evaluate' \
    --output_dir './eval_output/my_eval' \
    --model_name_or_path 'facebook/contriever' \
    --encoder_class 'default' \
    --pooling 'mean' \
    --normalize 'no' \
    --query_max_len 256 \
    --passage_max_len 256 \
    --per_device_eval_batch_size 128 \
    --precompute_corpus_embs 'true' \
    --per_device_matmul_batch_size 40960 \
    --encoding_cache_dir './encoding_cache_root' \
    --cleanup_temp_artifacts 'false' \
    --dataloader_num_workers 4 \
    --broadcast_output 'false' \
    --trove_logging_mode 'local_main' \
    --pbar_mode 'main' \
    --print_mode 'main' \
    --fair_sharding 'false' \
    --report_to 'wandb' \
    --save_eval_topk_logits 'false'
```

Trove supports evaluation in distributed environments using multiple GPUs or even across multiple nodes.
The following command runs the above evaluation on a machine with 2 GPUs:

```bash
deepspeed --include localhost:0,1 inference.py \
    {rest of command line arguments are the same as above}
```

## Hard Negative Mining

Evaluation and hard negative mining pipelines are very similar using Trove.
So we use the same script as above (`inference.py`) to mine hard negatives.
We handle the small differences between evaluation and hard negative mining inside the script.
We also need to change a few command line arguments.

```bash
python inference.py \
    --job 'mine_hn' \
    --output_dir './hn_mining_output/my_run' \
    --model_name_or_path 'facebook/contriever' \
    --encoder_class 'default' \
    --pooling 'mean' \
    --normalize 'no' \
    --query_max_len 32 \
    --passage_max_len 128 \
    --per_device_eval_batch_size 128 \
    --precompute_corpus_embs 'true' \
    --per_device_matmul_batch_size 40960 \
    --encoding_cache_dir './encoding_cache_root' \
    --cleanup_temp_artifacts 'false' \
    --dataloader_num_workers 4 \
    --broadcast_output 'false' \
    --search_topk 15 \
    --no_annot_in_mined_hn 'true' \
    --merge_mined_qrels 'true' \
    --trove_logging_mode 'local_main' \
    --pbar_mode 'main' \
    --print_mode 'main' \
    --fair_sharding 'false' \
    --report_to 'none'
```

Similar to evaluation, you can run hard negative mining on multiple nodes and GPUs.
