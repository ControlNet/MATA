from typing import TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from ...execution.image_patch import ImagePatch


class States:
    @dataclass
    class Initial:
        image_patch: "ImagePatch"
        query: str

    @dataclass
    class StepwiseReasoning:
        pass

    @dataclass
    class OneshotReasoning:
        pass

    @dataclass
    class Answering:
        pass

    @dataclass
    class Specialized:
        pass

    @dataclass
    class Final:
        success: bool
        final_answer: str | list["ImagePatch"] | None

    @dataclass
    class Failure:
        feedback: str


State = (
    States.Initial
    | States.StepwiseReasoning
    | States.OneshotReasoning
    | States.Answering
    | States.Specialized
    | States.Final
    | States.Failure
)
