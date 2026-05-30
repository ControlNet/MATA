from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .code_generator import CodeGenerator
from .instruction_generator import InstructionGenerator
from ..verifier import CodeVerifier
from ...execution.python import PythonSession
from ...execution.image_patch import ImagePatch
from ...memory.shared_memory import SharedMemory
from ...prompt.stepwise.code_task import (
    gqa_task_desc,
    refcoco_task_desc,
)


class InstructionPlanner:

    @dataclass
    class Success:
        instruction: str

    @dataclass
    class Failure:
        feedback: str

    Result = Success | Failure

    def __init__(self, model_name: str, dataset: str, query: str, shared_memory: SharedMemory, num_instructions: int, num_trials: int):
        self.query = query
        self.shared_memory = shared_memory
        self.instruction_generator = InstructionGenerator(model_name, dataset, num_instructions, num_trials)
        self.reset()

    def reset(self):
        self.instructions = None
        self.probs = None

    async def __call__(self, current_step: int) -> Result:
        self.reset()
        self.instruction_generator.reset()
        self.instructions, self.probs = await self.instruction_generator(self.query, current_step, self.shared_memory)
        if len(self.instructions) == 0:
            return self.Failure("No instructions generated.")
        self.instructions = [instruction for _, instruction in sorted(zip(self.probs, self.instructions), key=lambda x: x[0], reverse=True)]
        self.probs = np.sort(self.probs)[::-1]
        instruction = self.instructions[int(np.argmax(self.probs))]
        return self.Success(instruction)


class StepwiseReasoner:

    @dataclass
    class Complete:
        result: list[ImagePatch] | str

    @dataclass
    class Incomplete:
        pass

    @dataclass
    class Failure:
        feedback: str

    Result = Complete | Incomplete | Failure

    def __init__(
        self,
        instruction_planner_model_name: str,
        code_model_name: str,
        dataset: str,
        image_patch: ImagePatch,
        query: str,
        num_instructions: int,
        num_instruction_generator_trials: int,
        num_code_generator_trials: int
    ) -> None:
        self.num_code_generator_trials = num_code_generator_trials
        self.code_generator = CodeGenerator(code_model_name, dataset)
        self.code_verifier = CodeVerifier()
        self.python_session = PythonSession(image_patch)
        self.image_patch = image_patch
        self.query = query
        self.shared_memory = self.image_patch.shared_memory
        self.instruction_planner = InstructionPlanner(
            model_name=instruction_planner_model_name,
            dataset=dataset,
            query=query,
            shared_memory=self.shared_memory,
            num_instructions=num_instructions,
            num_trials=num_instruction_generator_trials,
        )

        match dataset:
            case "gqa":
                self.task_desc = gqa_task_desc
            case "refcoco":
                self.task_desc = refcoco_task_desc
            case _:
                raise ValueError(f"Dataset {dataset} is not supported")

    async def __call__(self, current_step_index: int) -> Result:
        instruction_result = await self.instruction_planner(current_step_index)
        match instruction_result:
            case InstructionPlanner.Success(instruction):
                pass
            case InstructionPlanner.Failure(feedback):
                return self.Failure(feedback)
            case _:
                raise ValueError(f"Invalid instruction planner return: {instruction_result}")

        first_attempt_code_generator = True
        feedback = ""
        code: str | None = None
        result_verifier_return: PythonSession.Result | None = None
        for _ in range(self.num_code_generator_trials):
            # ----------------- Code Generation -----------------
            if first_attempt_code_generator:
                self.code_generator.reset()
                response = await self.code_generator(self.query, instruction, current_step_index + 1, self.shared_memory)
                if response is not None:
                    first_attempt_code_generator = False
                else:
                    continue
            else:
                response = await self.code_generator.feedback(feedback)
                if response is None:
                    feedback = "Failed to generate code."
                    continue

            # ----------------- Code Verifier -----------------
            assert response is not None, "Response is None"
            code_verifier_return = self.code_verifier(response)
            match code_verifier_return:
                case CodeVerifier.Success(c):
                    code = c.replace("image_patch = ImagePatch(image)", "")
                case CodeVerifier.Failure(fb):
                    feedback = fb
                    continue
                case _:
                    raise ValueError(f"Invalid code verifier return: {code_verifier_return}")
        
            # ----------------- Python Code Executor -----------------
            # Execute the code and verify the result
            result_verifier_return = self.python_session(code)

            match result_verifier_return:
                case PythonSession.Failure(traceback):
                    feedback = f"The code is incorrect. Please fix the code, and make sure the new version of the code is wrapped in <PythonCode> tags. \n{traceback}"
                    continue
                case PythonSession.WrongAnswerType():
                    feedback = f"The code is incorrect. Please fix the code, and make sure the new version of the code is wrapped in <PythonCode> tags, based on the task description:\n{self.task_desc}"
                    continue
                case PythonSession.Complete(final_answer):
                    return self.Complete(final_answer)
                case PythonSession.Incomplete():
                    return self.Incomplete()
                case _:
                    raise ValueError(f"Invalid python session return: {result_verifier_return}")
        
        return self.Failure(feedback)
