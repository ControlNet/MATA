from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class Verifier(ABC):
    @dataclass
    class Success:
        value: str

    @dataclass
    class Failure:
        feedback: str

    Result = Success | Failure

    @abstractmethod
    def __call__(self, response: str) -> Result:
        pass
