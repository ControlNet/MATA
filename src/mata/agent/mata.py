from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Type
from PIL import Image

from ..memory.shared_memory import SharedMemory
from ..util.config import Config
from ..util.console import logger

from ..execution.image_patch import ImagePatch
from .stepwise.agent import StepwiseReasoner
from .base import Agent
from .oneshot.agent import OneshotReasoner
from .specialized.agent import SpecializedAgent
from .answering.agent import Answerer
from .hyper_agent import AutoHyperAgent
from .automaton.state import State, States
from .automaton.transitions import state_transition_map

class MATA:
    """Hyper Automaton runtime loop for multi-agent visual reasoning."""

    @dataclass
    class Success:
        final_answer: str | list[ImagePatch]

    @dataclass
    class Incomplete:
        pass

    @dataclass
    class Failure:
        feedback: str

    Result = Success | Incomplete | Failure

    def __init__(self, image_path: str, query: str) -> None:
        image_pil = Image.open(image_path)
        self.image_path = image_path
        self.shared_memory = SharedMemory()
        try:
            self.image_patch = ImagePatch(image_pil, shared_memory=self.shared_memory)
        except TypeError:
            raise ImageLoadingException(f"Image {image_path} is failed to load")
        init_state = States.Initial(self.image_patch, query)
        self.context: dict[str, Any] = {
            "image_path": image_path,
            "stepwise_step": 0,
            "current_step": 0,
            "next_iterations": 0,
            "max_next_iterations": Config.base_config["max_next_iterations"],
            "max_transition_steps": Config.base_config["max_transition_steps"],
            "dataset_name": Config.base_config["dataset"],
            "query": query,
            "state": init_state,
            "state_history": [init_state]
        }
        self.stepwise_reasoner = StepwiseReasoner(
            instruction_planner_model_name=Config.base_config["llm_model"],
            code_model_name=Config.base_config["llm_code_model"],
            dataset=Config.base_config["dataset"],
            image_patch=self.image_patch,
            query=self.query,
            num_instructions=Config.base_config["stepwise_reasoner"]["instruction_planner"]["num_instructions"],
            num_instruction_generator_trials=Config.base_config["stepwise_reasoner"]["instruction_planner"]["num_trials"],
            num_code_generator_trials=Config.base_config["stepwise_reasoner"]["num_code_generator_trials"],
        )

        self.oneshot_reasoner = OneshotReasoner(
            image_patch=self.image_patch,
            query=self.query
        )
        
        self.answerer = Answerer(
            model_name=Config.base_config["llm_model"],
            dataset=Config.base_config["dataset"],
            query=self.query,
            shared_memory=self.shared_memory
        )
        
        self.specialized_agent = SpecializedAgent(
            image_patch=self.image_patch,
            query=self.query
        )

        self.hyper_agent = AutoHyperAgent(Config.base_config["hyper_agent"]["model_name"])

        self.feedbacks = []
        
    @property
    def query(self) -> str:
        return self.context["query"]
    
    @query.setter
    def query(self, query: str):
        self.context["query"] = query

    @property
    def state_history(self) -> list[State]:
        return self.context["state_history"]

    @property
    def state(self) -> State:
        return self.context["state"]
    
    @state.setter
    def state(self, state: State):
        self.context["state"] = state

    @property
    def current_step(self) -> int:
        return self.context["current_step"]
    
    @current_step.setter
    def current_step(self, current_step: int):
        self.context["current_step"] = current_step

    @property
    def next_iterations(self) -> int:
        return self.context["next_iterations"]
    
    @next_iterations.setter
    def next_iterations(self, next_iterations: int):
        self.context["next_iterations"] = next_iterations

    @property
    def max_next_iterations(self) -> int:
        return self.context["max_next_iterations"]
        
    @property
    def max_transition_steps(self) -> int:
        return self.context["max_transition_steps"]

    async def step(self) -> Result:
        if self.current_step >= self.max_transition_steps:
            return self.Failure("\n".join(self.feedbacks))
        # if it is error handling, we have to search the parent state of the failure state
        if isinstance(self.state, States.Failure):
            for state in reversed(self.state_history):
                if isinstance(state, (States.Initial, States.Answering, States.OneshotReasoning)):
                    self.state = state
                    break
            else:
                self.state = self.state_history[0]
        # the type of the new state should be the next_state_type, but also may be Failure
        prev_state = self.state

        next_state_type = self.hyper_agent(self.shared_memory, self.context)
        if next_state_type == States.Failure:
            return self.Failure("\n".join(self.feedbacks))

        self.state = self.prepare_next_state(next_state_type)
        self.state_history.append(self.state)
        logger.debug(f"State: {prev_state.__class__.__name__} -> {self.state.__class__.__name__}")
        match self.state:
            case States.Specialized():
                try:
                    self.context["specialized_result"] = await self.specialized_agent()
                except TypeError:
                    raise ImageLoadingException(f"Image {self.image_path} is failed to load")
            case States.Final(True, final_answer) if final_answer is not None:
                return self.Success(final_answer)
            case States.Final(False, _) | States.Final(_, None):
                return self.Failure("\n".join(self.feedbacks))
            case States.Failure(error_feedback):
                self.feedbacks.append(error_feedback)
                self.hyper_agent.record_failure(prev_state)
            case States.StepwiseReasoning():
                self.context["stepwise_reasoning_result"] = await self.stepwise_reasoner(self.context["stepwise_step"])
                self.context["stepwise_step"] += 1
            case States.Answering():
                match prev_state:
                    case States.StepwiseReasoning():
                        self.context["answering_result"] = await self.answerer(self.context["stepwise_reasoning_result"])
                    case States.OneshotReasoning():
                        self.context["answering_result"] = await self.answerer(self.context["oneshot_reasoning_result"])
                    case _:
                        raise ValueError(f"{prev_state} is not expected")
            case States.OneshotReasoning():
                self.context["oneshot_reasoning_result"] = await self.oneshot_reasoner(self.shared_memory)
            case _:
                raise NotImplementedError(f"State {self.state} is not expected")
        self.current_step += 1
        if self.state in state_transition_map[type(prev_state)]["next"]:
            self.next_iterations += 1
        return self.Incomplete()
    
    def prepare_next_state(self, next_state_type: Type[State]) -> State:
        match (self.state, next_state_type):
            case (States.Initial(_, _), States.Specialized):
                return States.Specialized()
            case (States.Specialized(), States.Final):
                match self.context["specialized_result"]:
                    case Agent.Success(final_answer):
                        return States.Final(True, final_answer)
                    case Agent.Failure(feedback):
                        return States.Failure(feedback)
                    case _:
                        raise ValueError(f"{self.context['specialized_result']} result is not expected")
            case (States.Initial(_, _), States.StepwiseReasoning):
                return States.StepwiseReasoning()
            case (States.StepwiseReasoning(), States.Answering):
                match self.context["stepwise_reasoning_result"]:
                    case StepwiseReasoner.Complete(_) | StepwiseReasoner.Incomplete():
                        return States.Answering()
                    case StepwiseReasoner.Failure(feedback):
                        return States.Failure(feedback)
                    case _:
                        raise ValueError(f"{self.context['stepwise_reasoning_result']} result is not expected")
            case (States.OneshotReasoning(), States.Final):
                match self.context["oneshot_reasoning_result"]:
                    case Agent.Success(final_answer):
                        return States.Final(True, final_answer)
                    case Agent.Failure(feedback):
                        return States.Failure(feedback)
                    case _:
                        raise ValueError(f"{self.context['oneshot_reasoning_result']} result is not expected")
            case (States.OneshotReasoning(), States.Answering):
                return States.Answering()
            case (States.Answering(), States.OneshotReasoning):
                return States.OneshotReasoning()
            case (States.Answering(), States.StepwiseReasoning):
                return States.StepwiseReasoning()
            case (States.Answering(), States.Specialized):
                return States.Specialized()
            case (States.Answering(), States.Final):
                match self.context["answering_result"]:
                    case Answerer.Completed(answer):
                        return States.Final(True, answer)
                    case Answerer.Incomplete():
                        return States.StepwiseReasoning()
                    case Answerer.Failure(feedback):
                        return States.Failure(feedback)
                    case _:
                        raise ValueError(f"{self.context['answering_result']} result is not expected")
            case (States.Initial(_, _), States.OneshotReasoning):
                return States.OneshotReasoning()
            case _:
                raise NotImplementedError(f"State transition from {self.state} to {next_state_type} is not implemented")
    
    @classmethod
    def required_models(cls) -> list[str]:
        """Required models for this agent."""
        # Include all necessary models for SoTA performance
        return list(set([
            Config.base_config["depth_model"],
            Config.base_config["grounding_model"],
            Config.base_config["vlm_model"],
            Config.base_config["vlm_caption_model"],
            Config.base_config["verify_property_model"]
        ]))


class ImageLoadingException(Exception):
    pass
