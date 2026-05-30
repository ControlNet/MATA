from ...util.config import Config
from ...util.console import logger

from .specialist import GroundingSpecialist, SpecialistProtocal, VQASpecialist
from .verifier import GroundingVerifier, VQAVerifier, VerifierProtocol
from ..base import Agent
from ...execution.image_patch import ImagePatch

class SpecializedAgent(Agent):
    specialist: SpecialistProtocal
    verifier: VerifierProtocol

    def __init__(self, image_patch: ImagePatch, query: str):
        super().__init__(image_patch, query)
        self.task = Config.base_config["task"]
        match self.task:
            case "grounding":
                self.model_name = Config.base_config["specialized_agent"]["grounding_model"]
                self.specialist = GroundingSpecialist(self.model_name, Config.base_config["specialized_agent"]["grounding_threshold"])
                self.verifier = GroundingVerifier()
            case "vqa":
                self.model_name = Config.base_config["specialized_agent"]["vqa_model"]
                self.specialist = VQASpecialist(self.model_name)
                self.verifier = VQAVerifier()
            case _:
                raise ValueError("Invalid task")

    async def __call__(self) -> Agent.Result:
        result = self.specialist.step(self.image_patch, self.query)
        verified = self.verifier.step(self.image_patch, self.query, result)
        if verified:
            return self.Success(final_answer=result)
        else:
            assert self.task == "grounding", "Triggered error for non-grounding task in Specialized Agent verifier."
            assert isinstance(self.specialist, GroundingSpecialist), "Triggered error for non-grounding task in Specialized Agent verifier."
            if self.specialist.threshold > 0:
                self.specialist.threshold = max(0, self.specialist.threshold - 0.05)
                logger.debug(f"Specialized Agent: Threshold decreased to {self.specialist.threshold}.")
                return await self()
            else:
                return self.Failure(feedback="Cannot find any object.")

    @classmethod
    def required_models(cls) -> list[str]:
        match Config.base_config["task"]:
            case "grounding":
                return [Config.base_config["specialized_agent"]["grounding_model"]]
            case "vqa":
                return [Config.base_config["specialized_agent"]["vqa_model"]]
            case _:
                raise ValueError("Invalid task")
