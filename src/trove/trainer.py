"""Modified transformers trainer class to support saving custom retriever models."""

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from transformers import Trainer, TrainingArguments
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.utils import logging

from .evaluation import IRMetrics
from .logging_utils import get_logger_with_config
from .modeling import PretrainedRetriever

logger = logging.get_logger(__name__)


@dataclass
class RetrievalTrainingArguments(TrainingArguments):
    trove_logging_mode: str = "all"
    """Determines which processes can use the logging module. It is just a soft limit:
    the excluded processes can still log messages but their logging level is set to `ERROR`.
    You can choose from one of ``['main', 'local_main', and 'all']`` values.
    """

    def __post_init__(self):
        super().__post_init__()

        # set 'ddp_find_unused_parameters' to False if gradient checkpointing is enabled
        # in transformers.Trainer it is only done for instances of PreTrainedModel
        # The related snippet is here: https://github.com/huggingface/transformers/blob/f5620a76344595dbc7c9cff97bbd1edc1696854d/src/transformers/trainer.py#L2046
        # Since we are not an instance of PreTrainedModel, we should do it ourselves
        if self.gradient_checkpointing:
            # Make sure `ddp_find_unused_parameters` is not set to something that contradicts what we want to do
            # It is to safe guard against future changes in transformers inner workings
            assert self.ddp_find_unused_parameters in [None, False]
            self.ddp_find_unused_parameters = False

        if self.trove_logging_mode not in ["main", "local_main", "all"]:
            msg = "Valid values for mode configs (print and pbar) are ['main', 'local_main', and 'all']"
            raise ValueError(msg)

        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        if world_size < 2:
            return
        log_level = None
        if (self.trove_logging_mode == "main" and self.process_index != 0) or (
            self.trove_logging_mode == "local_main" and self.local_process_index != 0
        ):
            log_level = "ERROR"
        get_logger_with_config(
            name="trove", log_level=log_level, rank=self.process_index, force=True
        )


class RetrievalTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # https://github.com/texttron/tevatron/blob/7d298b428234f1c1065e98244827824753361815/src/tevatron/retriever/trainer.py#L18
        self.__is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.__is_ddp else 1

        # Chech if the passed compute_metrics class supports the given training arguments
        if self.compute_metrics is not None and isinstance(
            self.compute_metrics, IRMetrics
        ):
            self.compute_metrics.check_training_arguments(args=self.args)

        if self.train_dataset is not None and hasattr(
            self.train_dataset, "set_trainer"
        ):
            self.train_dataset.set_trainer(self)

    def training_step(self, *args, **kwargs):
        # https://github.com/texttron/tevatron/blob/7d298b428234f1c1065e98244827824753361815/src/tevatron/retriever/trainer.py#L53C66-L53C96
        return super().training_step(*args, **kwargs) / self._dist_loss_scale_factor

    def compute_loss(self, *args, **kwargs):
        super_output = super().compute_loss(*args, **kwargs)
        if "model" in kwargs:
            model = kwargs["model"]
        else:
            model = args[0]
        if model.training:
            if isinstance(super_output, tuple):
                return (super_output[0] * self._dist_loss_scale_factor, super_output[1])
            else:
                return super_output * self._dist_loss_scale_factor
        else:
            return super_output

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Saving model checkpoint to {output_dir}")

        if isinstance(self.model, PretrainedRetriever):
            # https://github.com/texttron/tevatron/blob/7d298b428234f1c1065e98244827824753361815/src/tevatron/retriever/trainer.py#L33
            if state_dict is None:
                state_dict = self.model.state_dict()
            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.args.save_safetensors,
            )
        elif isinstance(self.accelerator.unwrap_model(self.model), PretrainedRetriever):
            if state_dict is None:
                state_dict = self.model.state_dict()
            self.accelerator.unwrap_model(self.model).save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.args.save_safetensors,
            )
        else:
            msg = f"Only support 'PretrainedRetriever' models but got: {self.model.__class__.__name__}"
            raise RuntimeError(msg)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
