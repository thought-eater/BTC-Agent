from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Tuple


def run_jobs_serial(jobs: List[Tuple[str, Callable]]) -> dict:
    out = {}
    for job_id, fn in jobs:
        out[job_id] = fn()
    return out


def run_jobs_parallel(jobs: List[Tuple[str, Callable]], max_workers: int = 2) -> dict:
    if max_workers <= 1 or len(jobs) <= 1:
        return run_jobs_serial(jobs)

    out = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(fn): job_id for job_id, fn in jobs}
        for fut in as_completed(future_map):
            job_id = future_map[fut]
            out[job_id] = fut.result()
    return out
