import json
from pathlib import Path
from typing import Dict, Optional

import huggingface_hub as hf_hub
import torch
import torch.distributed as dist
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoModel, BitsAndBytesConfig, TrainingArguments

from .model_args import ModelArguments

# file that holds the trove retriever config and is saved next to weight checkpoints
TROVE_RETRIEVER_CONFIG_FILE = "trove_retriever_config.json"

# A subset of transformers.PretrainedModel attributes and methods to expose in wrappers
# to remain Compatible with a *subset* (and not all) of tools expecting a PreTrainedModel
DELEGATE_ATTRS_TO_WRAPPED = [
    "config",
    "is_loaded_in_4bit",
    "is_loaded_in_8bit",
    "gradient_checkpointing_enable",
    "gradient_checkpointing_disable",
    "enable_input_require_grads",
]


def add_model_apis_to_wrapper(wrapper, model) -> None:
    """Exposes some transformers model APIs through its wraper.

    This function exposes a subset of ``transformers.PretrainedModel`` attributes and methods in wrappers
    to remain Compatible with a **subset** (and not all) of tools expecting a PreTrainedModel like ``transformers.Trainer`` module.

    The added methods and attributes point to a method or attribute of the wrapped model with the same.

    Args:
        wrapper: the model that is wrapping a transformers model.
        model: an attribute of the ``wrapper`` class (often a transformers model).
    """
    global DELEGATE_ATTRS_TO_WRAPPED

    for attr in DELEGATE_ATTRS_TO_WRAPPED:
        if hasattr(model, attr):
            setattr(wrapper, attr, getattr(model, attr))


def load_trove_retriever_config(
    model_name: str, model_revision: Optional[str] = None
) -> Optional[Dict]:
    """Load the trove retriever config if present."""
    if Path(model_name).exists():
        # Model is saved locally
        config_path = Path(model_name).joinpath(TROVE_RETRIEVER_CONFIG_FILE)
        if not config_path.exists():
            return None
        assert config_path.exists()
    else:
        # Model is saved on hf hub
        assert hf_hub.repo_exists(repo_id=model_name)
        if not hf_hub.file_exists(
            repo_id=model_name,
            revision=model_revision,
            filename=TROVE_RETRIEVER_CONFIG_FILE,
        ):
            return None
        config_path = hf_hub.hf_hub_download(
            repo_id=model_name,
            revision=model_revision,
            filename=TROVE_RETRIEVER_CONFIG_FILE,
        )
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def find_base_name_or_path(model_name: str) -> Optional[str]:
    """Read the base model's name from model_name's config file."""
    if Path(model_name).exists():
        # Model is saved locally
        config_path = Path(model_name).joinpath("config.json")
        base_key = "_name_or_path"
        if not config_path.exists():
            # it is probably a peft model
            config_path = Path(model_name).joinpath("adapter_config.json")
            base_key = "base_model_name_or_path"
        assert config_path.exists()
    else:
        # Model is saved on hf hub
        assert hf_hub.repo_exists(repo_id=model_name)
        fname = "config.json"
        base_key = "_name_or_path"
        if not hf_hub.file_exists(repo_id=model_name, filename=fname):
            # it is probably a peft model
            fname = "adapter_config.json"
            base_key = "base_model_name_or_path"
        assert hf_hub.file_exists(repo_id=model_name, filename=fname)
        config_path = hf_hub.hf_hub_download(repo_id=model_name, filename=fname)

    with open(config_path, "r") as f:
        config = json.load(f)
    base_model_name = config.get(base_key, None)
    return base_model_name


def gather_tensors(
    tensor_obj: Optional[torch.Tensor], world_size: Optional[int], rank: Optional[int]
) -> torch.Tensor:
    """Gather tensors from all the processes in a distributed environment.

    see `this <https://github.com/texttron/tevatron/blob/7d298b428234f1c1065e98244827824753361815/src/tevatron/retriever/modeling/encoder.py#L99>`_

    Args:
        tensor_obj (Optional[torch.Tensor]): the tensor object to be gathered.
        world_size (Optional[int]): world size in the distributed environment
        rank (Optional[int]): rank of the process in the distributed environment

    Returns:
        The tensors across all processes concatenated.
    """
    if tensor_obj is None:
        return None

    tensor_obj = tensor_obj.contiguous()
    all_tensors = [torch.empty_like(tensor_obj) for _ in range(world_size)]
    dist.all_gather(all_tensors, tensor_obj)

    # Not sure, but I think this allows gradients to pass through:
    all_tensors[rank] = tensor_obj
    all_tensors = torch.cat(all_tensors, dim=0)
    return all_tensors


def load_adapter_config(
    model_name_or_path: str,
    revision: Optional[str] = None,
) -> Optional[Dict]:
    """Load peft adapter config from checkpoint path and returns None if adapter config file does
    not exist."""
    if Path(model_name_or_path).exists():
        # it is a path to a checkpoint
        adapter_path = Path(model_name_or_path) / "adapter_config.json"
        if not adapter_path.exists():
            return None
    else:
        # it is a model on hf hub.
        assert hf_hub.repo_exists(repo_id=model_name_or_path)
        if not hf_hub.file_exists(
            repo_id=model_name_or_path,
            filename="adapter_config.json",
            revision=revision,
        ):
            return None
        adapter_path = hf_hub.hf_hub_download(
            repo_id=model_name_or_path,
            revision=revision,
            filename="adapter_config.json",
        )
    with open(adapter_path, "r") as f:
        adapter_config = json.load(f)
    return adapter_config


def load_transformers_model(
    model_args: ModelArguments,
    training_args: Optional[TrainingArguments] = None,
    **kwargs,
):
    """Loads a huggingface transformers model.

    If the given checkpoints are only adapter weights, the adapters are merged into the main model.
    If specified by ``model_args`` attributes, this function quantizes the model and adds LORA adapters.

    Args:
        model_args: specifies how to load the model.
        training_args: It is used to setup gradient checkpointing arguments.
        **kwargs: passed to ``transformers.AutoModel.from_pretrained``.

    Returns:
        An instance of huggingface transformers model. Optionally with adapters.
    """
    if model_args.load_in_4bit or model_args.load_in_8bit:
        quant_conf = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            bnb_4bit_compute_dtype=model_args.torch_dtype,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
            bnb_4bit_quant_storage=model_args.torch_dtype,
        )
    else:
        quant_conf = None

    adapter_config = load_adapter_config(
        model_name_or_path=model_args.model_name_or_path,
        revision=model_args.model_revision,
    )

    if adapter_config is None:
        base_name_or_path = model_args.model_name_or_path
        base_rev = model_args.model_revision
        adapter_name_or_path = None
        adapter_rev = None
    else:
        if model_args.use_peft:
            raise NotImplementedError
        base_name_or_path = adapter_config["base_model_name_or_path"]
        base_rev = adapter_config["revision"]
        adapter_name_or_path = model_args.model_name_or_path
        adapter_rev = model_args.model_revision

    init_kwargs = dict(trust_remote_code=model_args.trust_remote_code)
    if model_args.attn_implementation is not None:
        init_kwargs["attn_implementation"] = model_args.attn_implementation
    if model_args.torch_dtype is not None:
        torch_dtype = model_args.torch_dtype
        if isinstance(torch_dtype, str) and torch_dtype != "auto":
            torch_dtype = getattr(torch, torch_dtype)
        init_kwargs["torch_dtype"] = torch_dtype

    if quant_conf is not None:
        init_kwargs["quantization_config"] = quant_conf

    model = AutoModel.from_pretrained(
        base_name_or_path, revision=base_rev, **init_kwargs, **kwargs
    )
    if adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(
            model,
            adapter_name_or_path,
            revision=adapter_rev,
            trust_remote_code=model_args.trust_remote_code,
        )
        model = model.merge_and_unload()

    if model_args.use_peft and quant_conf is not None:
        prep_kwargs = dict()
        if training_args is not None:
            prep_kwargs["use_gradient_checkpointing"] = (
                training_args.gradient_checkpointing
            )
            prep_kwargs["gradient_checkpointing_kwargs"] = (
                training_args.gradient_checkpointing_kwargs
            )
        model = prepare_model_for_kbit_training(model, **prep_kwargs)
    elif model_args.use_peft and getattr(
        training_args, "gradient_checkpointing", False
    ):
        model.enable_input_require_grads()

    if model_args.use_peft:
        target_modules = model_args.lora_target_modules
        if not isinstance(target_modules, list):
            target_modules = target_modules.split(",")

        lora_config = LoraConfig(
            base_model_name_or_path=model_args.model_name_or_path,
            task_type=getattr(TaskType, model_args.lora_task_type),
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)

    return model


# taken from here:
# https://github.com/texttron/tevatron/blob/7afa8067b958ca68689cb3e23792034f52ded656/src/tevatron/retriever/modeling/dense.py
def first_token_pooling(last_hidden_state: torch.Tensor, **kwargs) -> torch.Tensor:
    reps = last_hidden_state[:, 0]
    return reps


# taken from here:
# https://github.com/texttron/tevatron/blob/7afa8067b958ca68689cb3e23792034f52ded656/src/tevatron/retriever/modeling/dense.py
def last_token_pooling(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        reps = last_hidden_state[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        reps = last_hidden_state[
            torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths
        ]
    return reps


# taken from here:
# https://github.com/texttron/tevatron/blob/7afa8067b958ca68689cb3e23792034f52ded656/src/tevatron/retriever/modeling/dense.py
def mean_pooling(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    masked_hiddens = last_hidden_state.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    return reps
