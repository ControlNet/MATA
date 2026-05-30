from ...execution.image_patch import ImagePatch
from typing import Protocol, Any


class VerifierProtocol(Protocol):
    def step(self, image_patch: ImagePatch, query: str, result: Any):
        ...


class GroundingVerifier(VerifierProtocol):
    def step(self, image_patch: ImagePatch, query: str, result: list):
        return len(result) > 0


class VQAVerifier(VerifierProtocol):
    def step(self, image_patch: ImagePatch, query: str, result: str):
        return True
