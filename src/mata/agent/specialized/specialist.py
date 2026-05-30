from typing import Protocol
from ...execution.image_patch import ImagePatch


class SpecialistProtocal(Protocol):
    model_name: str

    def step(self, image_patch: ImagePatch, query: str) -> list[ImagePatch] | str:
        ...


class GroundingSpecialist(SpecialistProtocal):
    def __init__(self, model_name: str, threshold: float):
        self.model_name = model_name
        self.threshold = threshold

    def step(self, image_patch: ImagePatch, query: str) -> list[ImagePatch]:
        return image_patch.find([query], box_threshold=self.threshold, model_name=self.model_name)[query]


class VQASpecialist(SpecialistProtocal):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def step(self, image_patch: ImagePatch, query: str) -> str:
        return image_patch.simple_query_vlm(query, self.model_name)
