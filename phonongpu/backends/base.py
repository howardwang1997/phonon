from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class ForceBackend(ABC):
    name = "base"

    def __init__(self, options=None):
        self.options = dict(options or {})

    @abstractmethod
    def evaluate(self, structure, evals, executor=None):
        raise NotImplementedError

    def force_callable(self, structure, evals, executor=None):
        forces = self.evaluate(structure, evals, executor)
        table = {}
        for (idx, vec), f in zip(evals, forces):
            table[(int(idx), tuple(np.round(vec, 10)))] = f

        def _call(idx, vec):
            key = (int(idx), tuple(np.round(np.asarray(vec), 10)))
            return table[key]
        return _call, forces
