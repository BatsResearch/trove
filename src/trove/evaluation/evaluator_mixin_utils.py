import inspect
import json
import os
import pickle as pkl
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
from uuid import uuid4

import accelerate
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from ..file_utils import JSONLinesWriter
from ..logging_utils import get_logger_with_config
from . import callback_wandb
from .evaluation_args import EvaluationArguments
from .infinite_barrier import InfiniteBarrier

logger, logging_conf = get_logger_with_config("trove")


class RetrievalEvaluatorUtilsMixin:
    def __init__(
        self,
        args: Optional[EvaluationArguments] = None,
        tracker_init_kwargs: Optional[Dict] = None,
        tracker_extra_configs: Optional[Union[List[Dict], Dict]] = None,
        tracker_callbacks: Optional[Union[Any, List[Any]]] = None,
    ) -> None:
        """A simple Mixin for ``RetrievalEvaluator`` that provides convenient utilities.

        Anything that is not part of the main algorithm or is not one of your design decisions should go here.
        For example:

            * dealing with distributed env attributes: is it main process, local process, etc.
            * providing progress bar in distributed environment
            * create a temp file that is shared among processes
            * saving/loading files and checkpoints (what you save and where you save it should be decided in the main class.
              You just define a function here that takes those as arguments and make sure it is written correctly. e.g., only on main process)
            * logging things to wandb, etc.

        But, even small details that impact your design should not be here.
        For example, if there is a temp file that you only delete it under specific circumstances,
        you should check whether to delete the file in the main class.

        Args:
            args (Optional[EvaluationArguments]): Evaluation arguments. Same as what is passed to ``RetrievalEvaluator``.
            tracker_init_kwargs (Optional[Dict]): extra kwargs for initializing experiment trackers.
                Directly passed to tracker init method with dict comprehension.
            tracker_extra_configs (Optional[Union[List[Dict], Dict]]): extra configs to log with experiment trackers.
                These configs are merged with the evaluator and model configs if present.
                You can either pass one config object or a list of config objects.
                Config objects must be instances of python dictionary.
            tracker_callbacks (Optional[Any]): One or multiple custom experiment tracker callbacks.
                Callbacks must have a ``.log()`` method for logging the evaluation metrics.
                and a ``.setup()`` method that is called in the Evaluator constructor to initialize the tracker.
                If in a distributed environment, these methods will only be called from the main global process.
        """
        self.accelerator_state = None
        self._is_distributed = None
        self._rank = None
        self._world_size = None
        self._distributed_info_up_to_date = False
        self.update_distributed_state()

        args._setup_devices

        self.args = args

        # for dist.barrier without timeout
        self._inf_barrier_obj = None
        self.init_infinite_barrier()

        if (
            not self.args.overwrite_output_dir
            and self.args.output_dir is not None
            and Path(self.args.output_dir).exists()
            and list(Path(self.args.output_dir).iterdir())
        ):
            msg = (
                "The given output directory is not empty."
                f" Given output_dir: '{self.args.output_dir}'"
            )
            raise RuntimeError(msg)

        self.tracker_init_kwargs = tracker_init_kwargs
        self.tracker_extra_configs = tracker_extra_configs

        if tracker_callbacks is None:
            self.tracker_callbacks = []
        elif isinstance(tracker_callbacks, list):
            self.tracker_callbacks = [i for i in tracker_callbacks]
        else:
            self.tracker_callbacks = [tracker_callbacks]

        if self.args.report_to is not None and len(self.args.report_to) > 0:
            if len(self.args.report_to) > 1 or self.args.report_to[0] != "wandb":
                other_trackers = [t for t in self.args.report_to if t != "wandb"]
                msg = f"Only 'wandb' experiment tracker is supported. Got: {other_trackers}"
                raise ValueError(msg)
            # If we get to this line, this assert statement is definitely true
            # But, it makes the code easier to read/understand and does not cost anything
            assert len(self.args.report_to) == 1 and self.args.report_to[0] == "wandb"
            self.tracker_callbacks.append(callback_wandb.WandbCallback)

        # Parse arguments to check if we should have a progress bar and echo to stdout in this process
        self.pbar_allowed = self.allowed_on_this_process(self.args.pbar_mode)
        self.print_allowed = self.allowed_on_this_process(self.args.print_mode)

    def update_distributed_state(self) -> None:
        """Set the attributes related to the distributed environment."""
        self.accelerator_state = accelerate.PartialState()
        self._is_distributed = dist.is_initialized()
        if self._is_distributed:
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            # If it is not distributed, there is ony one process and that is the main one
            self._rank = 0
            self._world_size = 1

        self._distributed_info_up_to_date = True

    def init_infinite_barrier(self) -> None:
        """Create an instance of ``InfiniteBarrier`` for as a ``dist.barrier()`` alternative
        without timeout."""
        if self.world_size < 2:
            # there is only one process. barrier is not needed
            return
        addr = os.environ["MASTER_ADDR"]
        port = self.get_free_port_on_master()
        self._inf_barrier_obj = InfiniteBarrier(
            master_addr=addr, port=port, world_size=self.world_size, rank=self.rank
        )

    def allowed_on_this_process(self, mode: Optional[str]) -> bool:
        """Parse user provided ``mode`` argument to determine if we are allowed to perform certain
        actions on this processes.

        For example, to check if we should use a progress bar or print print results to stdout.
        If not in a distributed environment, always returns True.
        Valie modes are:

            * ``None`` : all processes are allowed
            * ``one`` : No process is allowed
            * ``main`` : only main process is allowed
            * ``local_main`` : only local main process is allowed
            * ``all`` : all processes are allowed

        Args:
            mode (Optional[str]): mode of operation for a specific action.

        Returns:
            a boolean showing that if we can or cannot perform a specific action from the current process.
        """
        # Check the provided mode is valid
        if mode is not None and mode not in ["none", "main", "local_main", "all"]:
            msg = "Valid values for mode configs (print and pbar) are ['none', 'main', 'local_main', and 'all']"
            raise ValueError(msg)
        # If not distributed, we are the only processes and allowed to do everything
        if not self.is_distributed:
            return True
        if mode is not None and mode == "none":
            return False

        is_allowed = (
            mode is None
            or mode == "all"
            or (mode == "main" and self.is_main_process)
            or (mode == "local_main" and self.is_local_main_process)
        )
        return is_allowed

    def all_gather_object(self, obj: Any) -> List[Any]:
        """Gather a pickleable object from all processes in a distributed environment.

        If not in a distributed environment, it simulates a distributed environment with only one process
        and returns a list with only one item, which is the object from the current (and only) process.

        Args:
            obj (Any): The pickleable object to gather from all processes

        Returns:
            A list of objects. where ``len(objects) == self.world_size``
            and ``objects[rank]`` is the data collected from process with ``rank``
        """
        # We cannot gather None
        assert obj is not None
        if not self.is_distributed:
            # simulate distributed environment
            return [pkl.loads(pkl.dumps(obj))]
        obj_container = [None for _ in range(self.world_size)]
        dist.all_gather_object(obj_container, obj)
        return obj_container

    def gather_object(self, obj: Any, dst: int = 0) -> Optional[List[Any]]:
        """Same as ``all_gather_object()`` but only gathers objects on process with ``rank ==
        dst``.

        Args:
            obj (Any): see :meth:`all_gather_object`
            dst (int): rank of the process that collects the objects from the entire group.

        Returns:
            ``None`` if ``dst != self.rank``. Otherwise, a list of objects. where ``len(objects) == self.world_size``
            and ``objects[rank]`` is the data collected from process with ``rank``
        """
        # We cannot gather None
        assert obj is not None
        if not self.is_distributed:
            # simulate distributed environment
            return [pkl.loads(pkl.dumps(obj))] if self.rank == dst else None
        if self.rank == dst:
            obj_container = [None for _ in range(self.world_size)]
        else:
            obj_container = None
        dist.gather_object(obj=obj, object_gather_list=obj_container, dst=dst)
        return obj_container

    def broadcast_obj(self, obj: Any, src: int = 0) -> Any:
        """Broadcast a pickleable object to all processes.

        Broadcast the obj from process with ``rank == src`` to all processes.
        Returns the broadcasted object.
        I.e., in all processes, returns the object from process with ``rank == src``.

        Args:
            obj (Any): pickleable object to broadcast to all processes.
            src (int): rank of the process that broadcasts the object.

        Returns:
            the broadcasted object from process with ``rank == src``.
        """
        if not self.is_distributed:
            # simulate distributed environment
            return obj
        obj_list = [obj]
        dist.broadcast_object_list(obj_list, src=src)
        return obj_list[0]

    def barrier(self, infinite: bool = False) -> None:
        """Similar to ``dist.barrier()``.

        In future, it should call the appropriate method for different distributed computing
        environments

        Args:
            infinite: If True, use ``InfiniteBarrier`` which can wait forever without a timeout. Otherwise, use regular ``dist.barrier()``.
        """
        if self.world_size < 2:
            return

        if infinite:
            self._inf_barrier_obj.sync()
        else:
            dist.barrier()

    @property
    def device(self) -> torch.device:
        """Current device that model and data should be moved to."""
        return self.args.device

    @property
    def is_distributed(self) -> bool:
        """Returns true if running in a distributed environment."""
        return self._is_distributed

    @property
    def rank(self) -> int:
        """Global rank of current process.

        Returns 0 if not running in distributed environment.
        """
        return self._rank

    @property
    def world_size(self) -> int:
        """The total number of processes.

        Returns 1 if not running in a distributed environment.
        """
        return self._world_size

    @property
    def is_main_process(self) -> bool:
        """Returns true if this is the global main process."""
        return self.args.process_index == 0

    @property
    def is_local_main_process(self) -> bool:
        """Returns true if this is the local main process."""
        return self.args.local_process_index == 0

    def all_devices_are_similar(self) -> bool:
        """Returns true if all devices available in a distributed environment are the same and
        False otherwise."""
        if not self.is_distributed:
            return True
        if self.world_size == 1:
            return True

        # Cuda attributes for current device
        cuda_attrs = torch.cuda.get_device_properties(self.device)
        dev_attrs = dict(
            name=cuda_attrs.name,
            major=cuda_attrs.major,
            minor=cuda_attrs.minor,
            total_memory=cuda_attrs.total_memory,
            multi_processor_count=cuda_attrs.multi_processor_count,
        )
        # Cuda attributes of all devices in the environment
        all_dev_attrs = self.all_gather_object(obj=dev_attrs)

        # Attributes that should be identical to claim similar devices
        for attr in ["name", "major", "minor", "multi_processor_count"]:
            if len(set([d[attr] for d in all_dev_attrs])) != 1:
                return False
        # tolerate up to 1GB of difference in memory. Assume similar memory capacity if difference <=1GB.
        mem_0 = all_dev_attrs[0]["total_memory"]
        for d in all_dev_attrs[1:]:
            if abs(d["total_memory"] - mem_0) >= 2**30:
                return False
        return True

    def get_free_port_on_master(self) -> int:
        """Finds a free tcp port on master host and shares it with all processes."""
        import socket

        def is_port_in_use(port: int) -> bool:
            # returns True if port is in use and False otherwise
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(("localhost", port)) == 0

        port = 0
        if self.is_main_process:
            # we only need to look for the port on the main process
            _sd = (os.getpid() + int(1000 * time.time())) % 2**32
            _rs = np.random.RandomState(_sd)
            while True:
                _port = int(_rs.choice(range(8000, 9999)))
                if not is_port_in_use(_port):
                    port = _port
                    break
        if self.is_distributed:
            # share the port with other processes
            port = self.broadcast_obj(obj=str(port), src=0)
        port = int(port)
        return port

    def get_shared_uuid(self) -> str:
        """Generates a random UUID that is the same across processes in a distributed
        environment."""
        unique_id = str(uuid4().hex)
        if self.is_distributed:
            # If in a distributed environment,
            # use the random name generated by the master processes
            unique_id = self.broadcast_obj(obj=unique_id, src=0)
        return unique_id

    def pbar(self, *args, **kwargs) -> tqdm:
        """Returns a ``tqdm`` instance.

        It is disabled if pbars are not allowed in this process.
        """
        kwargs["disable"] = kwargs.get("disable", False) or not self.pbar_allowed
        kwargs["dynamic_ncols"] = True
        return tqdm(*args, **kwargs)

    def _print(self, *args, **kwargs) -> None:
        """Call the builtin ``print()`` function we are allowed to print to stdout from this
        process."""
        if self.print_allowed:
            print(*args, **kwargs)

    def initialize_trackers(self) -> None:
        """Initialize experiment trackers."""
        if not self.is_main_process:
            return
        if not self.tracker_callbacks:
            return

        _run_name = None
        if (
            self.args.run_name is not None
            and self.args.run_name != self.args.output_dir
        ):
            _run_name = self.args.run_name

        for i in range(len(self.tracker_callbacks)):
            # instantiate trakcker callbacks if they are not already instantiated
            if inspect.isclass(self.tracker_callbacks[i]):
                tracker_cls = self.tracker_callbacks[i]
                self.tracker_callbacks[i] = tracker_cls()
            self.tracker_callbacks[i].setup(
                run_name=_run_name,
                args=self.args,
                model=self.model,
                init_kwargs=self.tracker_init_kwargs,
                extra_configs=self.tracker_extra_configs,
            )

    def log_metrics(self, metrics: Dict) -> None:
        """Log metrics to experiment trackers."""
        if not self.is_main_process:
            return
        if not self.tracker_callbacks:
            return
        for tracker in self.tracker_callbacks:
            tracker.log(metrics)

    def write_json(self, obj: Any, path: os.PathLike, **kwargs) -> None:
        """Write json file only from the main process.

        Args:
            obj: json serializable object to write to json file.
            path: path to destination file.
            **kwargs: keyword arguments passed to ``json.dump()``.
        """
        if not self.is_main_process:
            return
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            json.dump(obj, f, **kwargs)

    def write_json_lines(
        self, obj_list: Iterable[Any], path: os.PathLike, **kwargs
    ) -> None:
        """Write json lines file only from the main process.

        Args:
            obj_list: list of json serializable objects to write to json file.
            path: path to destination file.
            chunk_size: if provided, force flush the file buffer every ``chunk_size`` records.
            **kwargs: keyword arguments passed to ``json.dumps()``.
        """
        if not self.is_main_process:
            return
        with JSONLinesWriter(path=path, **kwargs) as writer:
            writer.add(obj_list)
