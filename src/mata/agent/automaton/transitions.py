from __future__ import annotations
from typing import Type
from .state import State, States


state_transition_map: dict[Type[State], dict[str, list[Type[State]]]] = {
    States.Initial: {
        "normal": [States.Specialized, States.OneshotReasoning, States.StepwiseReasoning],
        "next": [],
    },
    States.Specialized: {
        "normal": [States.Final],
        "next": [],
    },
    States.OneshotReasoning: {
        "normal": [States.Final, States.Answering],
        "next": [],
    },
    States.StepwiseReasoning: {
        "normal": [States.Answering],
        "next": [],
    },
    States.Answering: {
        "normal": [States.Final],
        "next": [States.Specialized, States.OneshotReasoning, States.StepwiseReasoning],
    },
    States.Final: {
        "normal": [],
        "next": [],
    },
    States.Failure: {
        "normal": [],
        "next": [States.Specialized, States.OneshotReasoning, States.StepwiseReasoning],
    }
}
