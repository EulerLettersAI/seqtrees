from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def resolve_n_jobs(n_jobs: int | None) -> int:
    if n_jobs is None:
        return 1
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0. Use None, 1, a positive integer, or -1 for all cores.")
    if n_jobs < 0:
        return max(1, os.cpu_count() or 1)
    return n_jobs


def parallel_map(func: Callable[[T], R], items: Iterable[T], n_jobs: int | None) -> list[R]:
    values = list(items)
    workers = resolve_n_jobs(n_jobs)
    if workers == 1 or len(values) <= 1:
        return [func(value) for value in values]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(func, values))
