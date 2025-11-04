import sys
from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path
from typing import Optional

from mpire import WorkerPool

# Set default start method based on Python version.
# Python <3.12 has a hang issue when using spawn.
# In Python 3.12+, fork is deprecated.
if (sys.version_info[0] == 3) and (sys.version_info[1] >= 12):
    DEFAULT_START_METHOD = "spawn"
else:
    DEFAULT_START_METHOD = "fork"


def worker_init(worker_state, reader_class, path: Path, backend: str) -> None:  # noqa
    """Worker initialization function for concurrent functions with reader."""
    worker_state["reader"] = reader_class(path, backend)


def prepare_worker_pool(
    reader,  # noqa
    worker_fn: Callable,
    iterable_of_args: Iterable,
    iterable_length: int,
    num_workers: int,
) -> tuple[Optional[WorkerPool], Iterable]:
    """Prepare worker pool and iterable."""
    if num_workers <= 1:
        return None, (worker_fn({"reader": reader}, *args) for args in iterable_of_args)
    # Prepare pool.
    init_fn = partial(
        worker_init,
        reader_class=reader.__class__,
        path=reader.path,
        backend=reader._backend.BACKEND_NAME,
    )
    pool = WorkerPool(
        n_jobs=num_workers,
        use_worker_state=True,
        start_method=DEFAULT_START_METHOD,
    )
    iterable_of_args = pool.imap(
        func=worker_fn,
        iterable_of_args=iterable_of_args,
        iterable_len=iterable_length,
        worker_init=init_fn,
    )
    return pool, iterable_of_args


def close_pool(pool: Optional[WorkerPool]) -> None:
    if pool is not None:
        pool.terminate()
