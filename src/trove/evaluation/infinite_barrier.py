import time
from datetime import timedelta

import torch.distributed as dist
from tqdm import tqdm

from ..logging_utils import get_logger_with_config

logger, logging_conf = get_logger_with_config("trove")


class InfiniteBarrier:
    def __init__(
        self,
        master_addr: str,
        port: int,
        world_size: int,
        rank: int,
        timeout: int = 600,
    ) -> None:
        """Alternative to ``dist.barrier()`` that does not timeout.

        Use it like this::

            inf_barrier = InfiniteBarrier(...) # craete the barrier
            inf_barrier.sync() # instead of dist.barrier()

        If for whatever reason, you are not able to change the timeout of the process group
        in distributed pytorch, you can use this class instead of ``dist.barrier()`` to avoid timeout errors.
        In general, you should write code that does not timeout in barrier but just in case!

        This class implements the barrier by a tcp key-value store. Each process that enters the barrier
        will increment the key of the active barrier by one. When the value of the key is equal to world size,
        the processes are released to continue.

        Args:
            master_addr: ip address of the master port.
            port: free port on master node to listen for messages from other processes.
            world_size: number of total processes involved.
            rank: global rank of current process.
            timeout: timeout in seconds for tcp store. You probably do not need to use this ever.
        """
        self.world_size = world_size
        self.process_index = rank
        # this increments by one for each new barrier block
        self.barrier_idx = -1
        if world_size > 1:
            self.kv_store = dist.TCPStore(
                master_addr, port, world_size, rank == 0, timedelta(seconds=timeout)
            )
        else:
            self.kv_store = None

    @property
    def _barrier_name(self) -> str:
        """Name of the current barrier in the key-value store."""
        return f"barrier_{self.barrier_idx}"

    def _enter_barrier(self) -> None:
        """Enter the key of the current barrier in the key-value store by 1 indicating that one
        more process has reached this barrier."""
        # you only enter this function once for each barrier in the code.
        # So if you enter this function, it is a new barrier with a new index and name
        self.barrier_idx += 1
        self.kv_store.add(self._barrier_name, 1)

    def _is_barrier_sync(self) -> bool:
        """Returns True if all processes have reached this barrier."""
        return int(self.kv_store.get(self._barrier_name)) == self.world_size

    def sync(self) -> None:
        """Blocks the process until all processes reach this part of the code."""
        if self.kv_store is None:
            # there is only process. No need for barrier
            return
        # after 20 minutes of waiting, print warning messages every 2 minutes.
        s = time.time()
        last_log = s

        # announce to other that one new process has entered the barrier
        self._enter_barrier()
        # wait for 0.2 seconds before checking again if all processes have reached this barrier
        t = 0.2
        # remain in loop until all processes enter this function
        while not self._is_barrier_sync():
            time.sleep(t)
            # next time sleep longer
            t *= 2
            if t > 20:
                # do not wait sleep more than 20 seconds
                t = 0.2
            c = time.time()
            if (c - s) >= 1200 and (c - last_log) >= 120:
                e_fmt = tqdm.format_interval(c - s)
                msg = f"Process have been waiting in barrier for 'elapsed_time: {e_fmt}'. Make sure it is expected."
                logger.warning(msg)
                last_log = c

        # make sure all processes exit the loop before returning.
        # If you return and the interpreter immediately exits but other processes has not
        # yet checked the kv store, that is gonna raise an exception
        # we just want to make sure all processes will be here soon (in probably less than a second)
        # and let torch handle the rest
        dist.barrier()
