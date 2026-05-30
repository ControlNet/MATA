from ...llm import llm_with_message
from ...memory.shared_memory import SharedMemory

from ...prompt.code_api import gqa_code_api, refcoco_code_api
from ...prompt.oneshot.code_example import gqa_code_example, refcoco_code_example
from ...prompt.oneshot.task import (
    gqa_task_desc,
    gqa_task_title,
    refcoco_task_desc,
    refcoco_task_title,
)


class CodeGenerator:
    def __init__(self, model_name: str, dataset: str):
        self.model_name = model_name
        self.dataset = dataset
        self.conversation = [
            {"role": "developer", "content": "You are a helpful assistant specializing in visual reasoning tasks. Your goal is to generate Python code that solves a visual reasoning query using the provided code API and examples."},
        ]

    def get_prompt(self, query: str, shared_memory: SharedMemory | None = None):
        match self.dataset:
            case "gqa":
                return gqa_generator_prompt(query, shared_memory)
            case "refcoco":
                return refcoco_generator_prompt(query, shared_memory)
            case _:
                raise ValueError(f"Dataset {self.dataset} not supported")

    async def __call__(self, query: str, shared_memory: SharedMemory | None = None) -> str | None:
        prompt = self.get_prompt(query, shared_memory)
        self.conversation.append({"role": "user", "content": prompt})
        response = await llm_with_message(self.model_name, self.conversation)
        return response
    
    async def feedback(self, feedback: str) -> str | None:
        self.conversation.append({"role": "user", "content": feedback})
        response = await llm_with_message(self.model_name, self.conversation)
        return response


def build_generator_prompt(query: str, task_title: str, task_desc: str, code_api: str, code_example: str, shared_memory: SharedMemory | None = None) -> str:
    if shared_memory is not None:
        feedbacks_prompt = shared_memory.feedbacks_prompt.strip("\n")
        extra_context = f"""Extra Context
----------------
<ExtraContext>
{feedbacks_prompt}
</ExtraContext>"""
    else:
        extra_context = ""
    
    return f"""
API Specification
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
{task_desc}
</TaskDescription>

Example for Reference
---------------------
Here is an example that illustrates the expected format and approach:
<Example>
{code_example}
</Example>

User Query
----------
This is the query you need to solve:
<Query>{query}</Query>

{extra_context}

Code Initialization
-------------------
An instance of the `ImagePatch` class is already provided. Use the following initialization code as the starting point:
<ExecutedCode>
image_patch = ImagePatch(image)
</ExecutedCode>

Instruction:
------------
Generate Python code that utilizes the provided API and initialization to solve the query enclosed within the <PythonCode></PythonCode> block. Ensure your solution follows the structure and style of the given example. Ensure the variable `final_answer` is assigned to the result of the query.
Output:
"""


def gqa_generator_prompt(query: str, shared_memory: SharedMemory | None = None) -> str:
    return build_generator_prompt(query, gqa_task_title, gqa_task_desc, gqa_code_api, gqa_code_example, shared_memory)


def refcoco_generator_prompt(query: str, shared_memory: SharedMemory | None = None) -> str:
    return build_generator_prompt(query, refcoco_task_title, refcoco_task_desc, refcoco_code_api, refcoco_code_example, shared_memory)
