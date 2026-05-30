from ...memory.shared_memory import SharedMemory
from ...util.config import Config
from ...util.console import logger

from ..base import Agent
from .generator import CodeGenerator
from ..verifier import CodeVerifier
from ...execution.python import PythonSession
from ...execution.image_patch import ImagePatch
from ...prompt.oneshot.task import gqa_task_desc, refcoco_task_desc

class OneshotReasoner(Agent):
    def __init__(self, image_patch: ImagePatch, query: str):
        super().__init__(image_patch, query)
        self.generator_model = Config.base_config["llm_code_model"]
        self.dataset = Config.base_config["dataset"]
        self.generator = CodeGenerator(self.generator_model, self.dataset)
        self.code_verifier = CodeVerifier()
        self.max_iterations = Config.base_config["oneshot_reasoner"]["max_iterations"]
        match self.dataset:
            case "gqa":
                self.task_desc = gqa_task_desc
            case "refcoco":
                self.task_desc = refcoco_task_desc
            case _:
                raise ValueError(f"Dataset {self.dataset} is not supported")

    async def __call__(self, shared_memory: SharedMemory | None = None) -> Agent.Result:
        feedback: str = ""
        first_iteration = True
        use_as_part_of_mata = shared_memory is not None

        for _ in range(self.max_iterations):
            # Generate code (first time or with feedback)
            if first_iteration:
                response = await self.generator(self.query, shared_memory)
            else:
                assert feedback != "", "Feedback is required"
                response = await self.generator.feedback(feedback)

            first_iteration = False
            
            if response is None:
                # just try again
                first_iteration = True
                logger.warning("Failed to generate code. Just try again.")
                feedback = "Failed to generate code."
                continue

            code_verifier_return = self.code_verifier(response)

            match code_verifier_return:
                case CodeVerifier.Success(code):
                    self.current_code = code.replace("image_patch = ImagePatch(image)", "")
                case CodeVerifier.Failure(fb):
                    feedback = fb
                    continue
                case _:
                    raise ValueError(
                        f"Invalid code verifier return: {code_verifier_return}")

            # Execute the code with a fresh Python session for the Oneshot Reasoner.
            python_session = PythonSession(self.image_patch, reset_shared_memory=True)
            # after result verifier
            result_verifier_return = python_session(self.current_code)
            # Update shared memory.
            if use_as_part_of_mata:
                shared_memory.extend_memory(
                    python_session.shared_memory.feedbacks,
                    python_session.shared_memory.codes,
                    python_session.shared_memory.instructions,
                    python_session.shared_memory.variables,
                    python_session.shared_memory.variable_names
                )
            match result_verifier_return:
                case PythonSession.Complete(result):
                    return self.Success(result)
                case PythonSession.Incomplete():
                    feedback = "The code is incomplete. Ensure the variable `final_answer` is assigned to the result of the query."
                    continue
                case PythonSession.Failure(traceback):
                    logger.debug(self.current_code)
                    feedback = f"The code is incorrect. Please fix the code. \n{traceback}"
                    continue
                case PythonSession.WrongAnswerType():
                    feedback = f"The code is incorrect. Please fix the code based on the task description:\n{self.task_desc}"
                    continue
                case _:
                    raise ValueError(
                        f"Invalid result verifier return: {result_verifier_return}")

        return self.Failure(feedback)

    @classmethod
    def required_models(cls) -> list[str]:
        return list(set([Config.base_config["depth_model"], Config.base_config["grounding_model"], Config.base_config["vlm_model"], Config.base_config["vlm_caption_model"], Config.base_config["verify_property_model"]]))
