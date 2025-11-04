import multiprocessing as mp
import sys
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Optional

# Set default start method based on Python version.
# Python <3.12 has a hang issue when using spawn.
# In Python 3.12+, fork is deprecated.
if (sys.version_info[0] == 3) and (sys.version_info[1] >= 12):
    DEFAULT_START_METHOD = "spawn"
else:
    DEFAULT_START_METHOD = "fork"


# Global state for worker processes (used instead of MPire's use_worker_state)
_worker_state = {}


def _worker_init(reader_class, path: Path, backend: str) -> None:  # noqa
    """Worker initialization function for concurrent functions with reader."""
    _worker_state["reader"] = reader_class(path, backend)


def _worker_fn_wrapper(worker_fn: Callable, args) -> any:
    """Wrapper to call worker function with worker state."""
    return worker_fn(_worker_state, *args)


def prepare_worker_pool(
    reader,  # noqa
    worker_fn: Callable,
    iterable_of_args: Iterable,
    iterable_length: int,
    num_workers: int,
) -> tuple[Optional[ProcessPoolExecutor], Iterable]:
    """Prepare worker pool and iterable."""
    if num_workers <= 1:
        return None, (worker_fn({"reader": reader}, *args) for args in iterable_of_args)

    # Prepare pool with multiprocessing context
    ctx = mp.get_context(DEFAULT_START_METHOD)
    pool = ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(reader.__class__, reader.path, reader._backend.BACKEND_NAME),
    )

    # Map function with wrapped worker function
    wrapped_fn = partial(_worker_fn_wrapper, worker_fn)
    iterable_of_args = pool.map(wrapped_fn, iterable_of_args)

    return pool, iterable_of_args


def close_pool(pool: Optional[ProcessPoolExecutor]) -> None:
    if pool is not None:
        pool.shutdown(wait=True)
