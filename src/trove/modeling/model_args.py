"""From TRL Library:

https://github.com/huggingface/trl/blob/d57e4b726561e5ae58fdc335f34029052944a4a3/trl/trainer/model_config.py
"""

from dataclasses import dataclass, fields
from typing import Dict, List, Optional


@dataclass
class ModelArguments:
    """Arguments which define the model and tokenizer to load."""

    model_name_or_path: Optional[str] = None
    """The model checkpoint for weights initialization."""
    model_revision: Optional[str] = None
    """The specific model version to use (can be a branch name, tag name or commit id)."""
    torch_dtype: Optional[str] = None
    """Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the
    dtype will be automatically derived from the model's weights.
    """
    trust_remote_code: bool = False
    """Trust remote code when loading a model."""
    attn_implementation: Optional[str] = None
    """Which attention implementation to use; you can run --attn_implementation=flash_attention_2, in which case you must install this manually by running `pip install flash-attn --no-build-isolation`"""
    encoder_class: Optional[str] = None
    """Name or alias of the PretrainedEncoder subclass that should be used as the encoder.
    If not specified, use the subclass of PretrainedEncoder that can load the given checkpoint.
    """
    pooling: Optional[str] = None
    """The type of pooling to use (options are 'mean', 'first_token', 'last_token'). Make sure the encoder class that you are using allows dynamically choosing the pooling method (i.e., supports setting this option)."""
    normalize: Optional[str] = None
    """Whether to normalize the embedding vector or not (options are 'yes' and 'no'). Make sure the encoder class that you are using allows selecting the normalization dynamically (i.e., supports setting this option)."""
    loss: Optional[str] = None
    """Name of the loss function to use for IR training."""
    temperature: float = 1.0
    """scaling factor for similarity scores when calculating the loss. We do not enforce its use.
    It is up to the loss function to use this argument.
    """
    temperature_learnable: bool = False
    """If true, make temperature a learnable parameter with initial value set to '--temperature' argument.
    WARNING: This feature is not complete yet and the learned temperature value is not saved to checkpoints. So you won't be able to load it later.
    """
    use_peft: bool = False
    """Whether to use PEFT or not for training."""
    lora_r: Optional[int] = 16
    """LoRA R value."""
    lora_alpha: Optional[int] = 32
    """LoRA alpha."""
    lora_dropout: Optional[float] = 0.05
    """LoRA dropout."""
    lora_target_modules: Optional[List[str]] = None
    """LoRA target modules."""
    lora_modules_to_save: Optional[List[str]] = None
    """Model layers to unfreeze & train"""
    lora_task_type: str = "FEATURE_EXTRACTION"
    """The task_type to pass for LoRA (use SEQ_CLS for reward modeling)"""
    use_rslora: bool = False
    """Use Rank-Stabilized LoRA (`paper <https://huggingface.co/papers/2312.03732>`_), which sets the adapter scaling factor to lora_alpha/√r, instead of the original default value of `lora_alpha/r`."""
    load_in_8bit: bool = False
    """use 8 bit precision for the base model - works only with LoRA"""
    load_in_4bit: bool = False
    """use 4 bit precision for the base model - works only with LoRA"""
    bnb_4bit_quant_type: Optional[str] = "nf4"
    """precise the quantization type (fp4 or nf4)"""
    use_bnb_nested_quant: bool = False
    """use nested quantization"""

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")

        if (
            isinstance(self.lora_target_modules, list)
            and len(self.lora_target_modules) == 1
        ):
            self.lora_target_modules = self.lora_target_modules[0]

        if self.use_rslora:
            raise NotImplementedError

        assert self.torch_dtype in ["auto", "bfloat16", "float16", "float32", None]
        assert self.normalize in ["yes", "no", None]

    def to_dict(self) -> Dict:
        """Return a json serializable view of the class attributes."""
        json_dict = dict()
        field_names = [f.name for f in fields(self)]
        for fname in field_names:
            fvalue = getattr(self, fname)
            json_dict[fname] = fvalue
        return json_dict
