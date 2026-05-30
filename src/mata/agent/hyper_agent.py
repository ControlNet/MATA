from abc import ABC, abstractmethod
from typing import Any, Type
import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

from ..memory.shared_memory import SharedMemory

from .automaton.state_controller_prompter import StateControllerPrompter
from .automaton.transitions import state_transition_map
from .automaton.state import State, States
from .verifier import XMLTagVerifier


_llm_state_controller = {
    "model": None,
    "tokenizer": None
}


class HyperAgent(ABC):
    @abstractmethod
    def __call__(self, shared_memory: SharedMemory, context: dict) -> Type[State]:
        pass


class AutoHyperAgent(HyperAgent):
    prompter = StateControllerPrompter()

    # Hyper Agent backed by the trained LLM State Controller.
    def __init__(self, model_name: str):
        # Mapping from state names to state types
        self.state_name_to_type = {
            "Initial": States.Initial,
            "StepwiseReasoning": States.StepwiseReasoning,
            "OneshotReasoning": States.OneshotReasoning,
            "Answering": States.Answering,
            "Specialized": States.Specialized,
            "Final": States.Final,
            "Failure": States.Failure
        }

        if _llm_state_controller["model"] is None:
            _llm_state_controller["model"] = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                device_map="auto",
            ).eval()
        self.model = _llm_state_controller["model"]

        if _llm_state_controller["tokenizer"] is None:
            _llm_state_controller["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = _llm_state_controller["tokenizer"]
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.failure_states = set()
        self.verifier = XMLTagVerifier(tag="NextState")
        self.prefix_fns = {} # mapping from state class names to prefix functions

    def get_state_candidates(self, context: dict) -> list[Type[State]]:
        state_next_candidates = state_transition_map[type(context["state"])]["next"]
        state_normal_candidates = state_transition_map[type(context["state"])]["normal"]

        if context["next_iterations"] >= context["max_next_iterations"]:
            next_state_candidates = state_normal_candidates
        else:
            next_state_candidates = [*state_normal_candidates, *state_next_candidates]

        next_state_candidates = [each for each in next_state_candidates if each.__name__ not in self.failure_states]
        return next_state_candidates
        
    def generate_prompt(self, shared_memory: SharedMemory, context: dict) -> list[dict[str, Any]]:
        next_state_candidates = self.get_state_candidates(context)
        return self.prompter(shared_memory, context, next_state_candidates, context["state_history"])
    
    def record_failure(self, failure_source_state: State):
        self.failure_states.add(failure_source_state.__class__.__name__)

    def get_prefix_fn(self, state_candidates: list[str]):
        regex = r"<NextState>(" + "|".join(state_candidates) + r")</NextState>"
        if regex in self.prefix_fns:
            return self.prefix_fns[regex]
        prefix_fn = build_transformers_prefix_allowed_tokens_fn(self.tokenizer, RegexParser(regex))
        self.prefix_fns[regex] = prefix_fn
        return prefix_fn
    
    def __call__(self, shared_memory: SharedMemory, context: dict) -> Type[State]:
        prompt = self.generate_prompt(shared_memory, context)

        state_candidates = [each.__name__ for each in self.get_state_candidates(context)]

        if len(state_candidates) == 0:
            return States.Failure

        if len(state_candidates) == 1:
            return self.state_name_to_type[state_candidates[0]]

        prefix_fn = self.get_prefix_fn(state_candidates)

        text = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=32768, # Limit generated tokens
                prefix_allowed_tokens_fn=prefix_fn
            )
        # Decode only the newly generated tokens
        outputs_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response_text = self.tokenizer.decode(outputs_ids, skip_special_tokens=True).strip("\n")

        # Extract predicted state
        pred_result = self.verifier(response_text)

        match pred_result:
            case self.verifier.Success():
                return self.state_name_to_type[pred_result.value]
            case self.verifier.Failure():
                return States.Failure
            case _:
                raise ValueError(f"Unknown prediction result: {pred_result}")
