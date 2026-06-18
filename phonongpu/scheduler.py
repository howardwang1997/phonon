from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def detect_gpus():
    if _HAS_TORCH and torch.cuda.is_available():
        return torch.cuda.device_count()
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        )
        idx = [int(x) for x in out.split()]
        if idx:
            return len(idx)
    except Exception:
        pass
    return 0


class MultiGPUExecutor:
    def __init__(self, num_gpus=None, jobs_per_gpu=1):
        self.num_gpus = num_gpus if num_gpus is not None else max(detect_gpus(), 1)
        self.jobs_per_gpu = jobs_per_gpu

    @property
    def max_workers(self):
        return self.num_gpus * self.jobs_per_gpu

    def gpu_ids(self, total):
        return [i % self.num_gpus for i in range(total)]

    def map(self, func, items, ordered=True):
        items = list(items)
        if not items:
            return []
        if self.max_workers <= 1:
            return [func(x) for x in items]
        results = [None] * len(items)
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            future_to_idx = {ex.submit(func, x): i for i, x in enumerate(items)}
            for fut in as_completed(future_to_idx):
                results[future_to_idx[fut]] = fut.result()
        return results

    def __repr__(self):
        return f"MultiGPUExecutor(num_gpus={self.num_gpus}, jobs_per_gpu={self.jobs_per_gpu})"
