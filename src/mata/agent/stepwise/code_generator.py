from __future__ import annotations

from ...llm import llm_with_message
from ...memory.shared_memory import SharedMemory
from ...util.config import Config

from ...prompt.stepwise.instruction_task import (
    gqa_task_title,
    refcoco_task_title,
    gqa_task_desc,
    refcoco_task_desc,
)
from ...prompt.stepwise.code_example import (
    gqa_code_example,
    refcoco_code_example,
)
from ...prompt.code_api import gqa_code_api, refcoco_code_api


class CodeGenerator:

    def __init__(self, model_name: str, dataset: str):
        self.model_name = model_name
        self.dataset = dataset
        self.reset()

    def reset(self):
        self.conversation = [
            {"role": "system", "content": "You are a helpful assistant specializing in visual reasoning tasks. Your goal is to generate Python code that solves a visual reasoning query using the provided code API and examples."}
        ]

    async def __call__(self, query: str, instruction: str, current_step_index: int, 
                       shared_memory: SharedMemory
                       ):
        prompt = self.build_prompt(
            query, instruction, current_step_index, shared_memory)

        if Config.debug:
            with open("stepwise_code_generator.txt", "w") as f:
                f.write(prompt)

        # Add prompt to conversation
        self.conversation.append({"role": "user", "content": prompt})

        # Generate code
        response = await llm_with_message(self.model_name, self.conversation)

        if response is not None:
            self.conversation.append({"role": "assistant", "content": response})
        return response

    async def feedback(self, feedback: str) -> str | None:
        self.conversation.append({"role": "user", "content": feedback})
        response = await llm_with_message(self.model_name, self.conversation)

        if response:
            self.conversation.append({"role": "assistant", "content": response})
        return response
    
    def build_prompt(self, query: str, instruction: str, current_step_index: int, shared_memory: SharedMemory) -> str:
        match self.dataset:
            case "gqa":
                task_title = gqa_task_title
                task_description = gqa_task_desc
                code_example = gqa_code_example
                code_api = gqa_code_api
            case "refcoco":
                task_title = refcoco_task_title
                task_description = refcoco_task_desc
                code_example = refcoco_code_example
                code_api = refcoco_code_api
            case _:
                raise ValueError(f"Dataset {self.dataset} is not supported")

        # Extract state information from memory bank
        previous_instructions = shared_memory.instructions_prompt.strip("\n")
        previous_code = shared_memory.codes_prompt.strip("\n")
        execution_results = shared_memory.feedbacks_prompt.strip("\n")
        variables_info = shared_memory.variables_prompt.strip("\n")

        # Format the prompt with clear sections
        prompt = f"""API Specification
-----------------
Use the following code API to guide your solution:
<CodeAPI>
{code_api}
</CodeAPI>

Task Description
----------------
Review the task description below to understand the problem context:
<TaskDescription>
{task_title}
{task_description}</TaskDescription>

Example Code
-----------
Here is an example that illustrates the expected format and approach:
<Examples>
{code_example}
</Examples>

User Query
----------
This is the query you need to solve:
<Query>{query}</Query>

Current Step
------------
<Step>{current_step_index}</Step>

Previous Instructions
---------------------
<PreviousInstructions>
{previous_instructions}
</PreviousInstructions>

Current Instruction
-------------------
<Instruction>{instruction}</Instruction>

Previously Executed Code
-----------------------
<ExecutedCode>
{previous_code}
</ExecutedCode>

Execution Results
----------------
<ExecutionResults>
{execution_results}
</ExecutionResults>

Available Variables
-------------------
<Variables>
{variables_info}
</Variables>

-------------------
Generate Python code that solves the query based on the current instruction.
Your code should build upon previous steps and use the available variables.
Use the code API as shown in the example.

Enclose your code in <PythonCode></PythonCode> tags.
If your code provides a final answer, assign it to a variable named 'final_answer'.
"""
        return prompt
