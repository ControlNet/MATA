from __future__ import annotations

from dataclasses import dataclass
import traceback
from typing import Any
from ..util.misc import get_statement_variable
from ..util.config import Config
from ..util.console import logger
from ..memory.shared_memory import SharedMemory

from .image_patch import ImagePatch


def get_description_from_executed_variable_list(executed_variable_list: list[str], local_variables: dict[str, Any]) -> list[str]:
    # run in executor only
    description = []
    for variable_name in executed_variable_list:
        one_variable = local_variables[variable_name]
        description.append(f'{variable_name}: {one_variable}')
        if isinstance(one_variable, ImagePatch):
            description[-1] += f', patch name: {one_variable.image_name}'
    return description


class PythonSession:
    """Python Executor with Result Verifier"""

    def __init__(self, image_patch: ImagePatch, reset_shared_memory: bool = False):
        self.globals = {}
        self.locals = {}
        self.image_patch = image_patch
        if not reset_shared_memory:
            self.shared_memory = image_patch.shared_memory
        else:
            self.shared_memory = SharedMemory()
        self.locals["image_patch"] = self.image_patch
        initial_code = "image_patch = ImagePatch(image)"
        self.shared_memory.codes.append(initial_code)
        self.shared_memory.variable_names.extend(get_statement_variable(initial_code, self.locals))
        self.shared_memory.variables.extend(get_description_from_executed_variable_list(self.shared_memory.variable_names, self.locals))
        exec("from mata.execution.image_patch import *", self.globals, self.locals)
        self.task = Config.base_config["task"]

    @dataclass
    class Complete:
        final_answer: list[ImagePatch] | str

    @dataclass
    class Incomplete:
        pass

    @dataclass
    class Failure:
        traceback: str

    @dataclass
    class WrongAnswerType:
        message: str

    Result = Complete | Incomplete | Failure | WrongAnswerType

    def __call__(self, code: str) -> PythonSession.Result:
        logger.debug(f"Executing code: {code}")
        try:
            exec(code, self.globals, self.locals)
        except Exception:
            return self.Failure(traceback.format_exc())

        new_variable_names = get_statement_variable(code, self.locals)
        new_variables = get_description_from_executed_variable_list(
            new_variable_names, self.locals)
        self.shared_memory.extend_memory(
            other_feedbacks=[], # already added when executing the image_patch code
            other_codes=[code],
            other_instructions=[],
            other_variables=new_variables,
            other_variable_names=new_variable_names
        )
        
        if "final_answer" not in self.locals:
            return self.Incomplete()
        
        final_answer = self.locals["final_answer"]
        match self.task:
            case "vqa":
                if isinstance(final_answer, str):
                    return self.Complete(final_answer)
                else:
                    self.shared_memory.return_final_answer_should_be_string()
                    return self.WrongAnswerType(
                        message=f"The final answer should be a string, but got {type(final_answer)}."
                    )
            case "grounding":
                if isinstance(final_answer, ImagePatch):
                    return self.Complete([final_answer])
                elif isinstance(final_answer, list) and all(isinstance(item, ImagePatch) for item in final_answer):
                    return self.Complete(final_answer)
                else:
                    self.shared_memory.return_final_answer_should_be_ImagePatch()
                    return self.WrongAnswerType(
                        message=f"The final answer should be an ImagePatch or a list of ImagePatch, but got {str(final_answer)}."
                    )
            case _:
                raise ValueError(f"Task {self.task} not supported.")
