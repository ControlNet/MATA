from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..execution.image_patch import ImagePatch


class Agent(ABC):

    @dataclass
    class Success:
        final_answer: str | list[ImagePatch]

    @dataclass
    class Failure:
        feedback: str

    Result = Success | Failure

    def __init__(self, image_patch: ImagePatch, query: str):
        self.query = query
        self.image_patch = image_patch
        self.shared_memory = self.image_patch.shared_memory

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Agent.Result:
        pass

    @classmethod
    @abstractmethod
    def required_models(cls) -> list[str]:
        pass
