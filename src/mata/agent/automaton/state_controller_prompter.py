from typing import Any, Type
from ...memory.shared_memory import SharedMemory

from .state import State, States
from ...prompt.oneshot.task import (
    gqa_task_title,
    gqa_task_desc,
    refcoco_task_title,
    refcoco_task_desc,
)

class StateControllerPrompter:
    state_descriptions = {
        States.Specialized: "Use a specialized visual model or task-specific expert to answer directly when the query can be handled by a dedicated specialist.",
        States.OneshotReasoning: "Generate and execute a complete one-shot reasoning program for the query, then return a final result directly or via answering.",
        States.StepwiseReasoning: "Run the Stepwise Reasoner. This state includes internal instruction planning, code generation, code execution, and memory updates for one reasoning step.",
        States.Answering: "Synthesize an answer from the visual evidence and variables accumulated in shared memory.",
        States.Final: "Terminate the run with a final answer or failure.",
        States.Failure: "Enter failure handling when the previous state cannot produce a valid result.",
    }

    def get_task_description(self, dataset_name: str) -> tuple[str, str]:
        match dataset_name:
            case "gqa":
                return gqa_task_title, gqa_task_desc
            case "refcoco":
                return refcoco_task_title, refcoco_task_desc
            case _:
                raise ValueError(f"Unknown dataset name: {dataset_name}")

    def __call__(self, shared_memory: SharedMemory, context: dict, next_state_candidates: list[Type[State]], state_history: list[State]) -> list[dict[str, Any]]:
        next_state_candidates_str = "\n".join([state.__name__ for state in next_state_candidates])
        next_state_descriptions_str = "\n".join([
            f"{state.__name__}: {self.state_descriptions.get(state, 'No additional description.')}"
            for state in next_state_candidates
        ])
        state_history_str = "\n".join([type(state).__name__ for state in state_history])
        task_title, task_description = self.get_task_description(context["dataset_name"])

        # System message explaining the task
        system_message = """You are an AI assistant to control the state of a multi-step visual reasoning system. Your task is to decide the next state the system should transition to based on the current state and history. """

        # Format user message with structured XML using shared memory properties
        user_message = f"""<TaskDescription>
{task_title}
{task_description}</TaskDescription>
        
<Query>
{context["query"]}
</Query>
        
<Instructions>
{shared_memory.instructions_prompt}
</Instructions>

<Feedback>
{shared_memory.feedbacks_prompt}
</Feedback>

<Code>
{shared_memory.codes_prompt}
</Code>

<Variables>
{shared_memory.variables_prompt}
</Variables>

<StateHistory>
{state_history_str}
</StateHistory>

<State>{type(context["state"]).__name__}</State>
<CurrentStep>{context["current_step"]}</CurrentStep>

Based on the information above, determine the next state the system should transition to. Choose from the following states:
<StateCandidates>
{next_state_candidates_str}
</StateCandidates>

<StateDescriptions>
{next_state_descriptions_str}
</StateDescriptions>

Return the name wrapped in <NextState> tags.
"""

        # {"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "..."}]}
        # Create huggingface SFT format
        result = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

        return result
