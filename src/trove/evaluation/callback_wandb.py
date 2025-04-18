import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Union

import wandb


class WandbCallback:
    def __init__(self) -> None:
        """Simple callback to log metrics to Wandb."""
        self.disabled = os.getenv("WANDB_DISABLED", "").upper() in {
            "1",
            "ON",
            "YES",
            "TRUE",
        }
        self.initialized = False
        self.wandb_run = None

    def setup(
        self,
        run_name: Optional[str] = None,
        args: Optional[Any] = None,
        model: Optional[Any] = None,
        init_kwargs: Optional[Dict] = None,
        extra_configs: Optional[Union[Dict, List[Dict]]] = None,
        **kwargs,
    ) -> None:
        """init a wandb run.

        Args:
            run_name (Optional[str]): human readable run name.
            args (Optional[Any]): instance of ``RetrievalEvaluator.args``
            model (Optional[Any]): model that is being evaluated
            init_kwargs (Optional[Dict]): additional kwargs for ``wandb.init()`` method
            extra_configs (Optional[Union[Dict, List[Dict]]]): additional configs to be merged with experiment config when initializing wandb.
            **kwargs: Not used. Just to swallow extra arguments.
        """
        if self.disabled or self.initialized:
            return
        self.initialized = True

        if init_kwargs is None:
            init_kwargs = dict()
        if "name" not in init_kwargs and run_name is not None:
            init_kwargs["name"] = run_name

        if "project" not in init_kwargs:
            init_kwargs["project"] = os.getenv("WANDB_PROJECT", "trove")

        if args is None:
            init_config = dict()
        else:
            init_config = {**args.to_dict()}

        if model is not None:
            if hasattr(model, "args") and model.args is not None:
                if isinstance(model.args, dict):
                    init_config = {**model.args, **init_config}
                elif hasattr(model.args, "to_dict"):
                    init_config = {**model.args.to_dict(), **init_config}
                else:
                    if is_dataclass(model.args):
                        args_dict = asdict(model.args)
                        init_config = {**args_dict, **init_config}
            elif hasattr(model, "config") and model.config is not None:
                model_config = (
                    model.config
                    if isinstance(model.config, dict)
                    else model.config.to_dict()
                )
                init_config = {**model_config, **init_config}

        if extra_configs is not None:
            if isinstance(extra_configs, dict):
                init_config = {**extra_configs, **init_config}
            elif isinstance(extra_configs, list):
                for conf in extra_configs:
                    init_config = {**conf, **init_config}
            else:
                msg = f"tracker_extra_configs should either be a dict or a list of dicts. Got: {type(extra_configs)}"
                raise TypeError(msg)

        self.wandb_run = wandb.init(**init_kwargs, config=init_config)

    def log(self, *args, **kwargs):
        """Log results to wandb run."""
        if self.disabled:
            return
        return self.wandb_run.log(*args, **kwargs)
