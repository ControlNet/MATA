from __future__ import annotations

from ...llm import llm_with_message
from ...memory.shared_memory import SharedMemory
from ...util.config import Config

from ...prompt.oneshot.task import (
    gqa_task_title,
    gqa_task_desc,
    refcoco_task_title,
    refcoco_task_desc,
)

class Summarizer:

    def __init__(self, model_name: str, dataset: str):
        self.model_name = model_name
        self.dataset = dataset
        self.conversation = []
        self.reset()

    def reset(self):
        self.conversation = [
            {"role": "system", "content": "You are an AI assistant designed to assist with compositional visual reasoning tasks by answering questions based on the available visual information."}
        ]

    async def __call__(self, query: str, shared_memory: SharedMemory) -> str | None:
        prompt = self.build_prompt(query, shared_memory)

        if Config.debug:
            with open("summarizer.txt", "w") as f:
                f.write(prompt)

        # Add prompt to conversation
        self.conversation.append({"role": "user", "content": prompt})

        # Get response
        response = await llm_with_message(self.model_name, self.conversation)

        if response is not None:
            self.conversation.append({"role": "assistant", "content": response})

        return response

    async def feedback(self, feedback: str) -> str | None:
        self.conversation.append({"role": "user", "content": feedback})
        response = await llm_with_message(self.model_name, self.conversation)

        if response is not None:
            self.conversation.append({"role": "assistant", "content": response})
        return response
    
    def build_prompt(self, query: str, shared_memory: SharedMemory) -> str:
        match self.dataset:
            case "gqa":
                task_title = gqa_task_title
                task_description = gqa_task_desc
            case "refcoco":
                task_title = refcoco_task_title
                task_description = refcoco_task_desc
            case _:
                raise ValueError(f"Unsupported dataset: {self.dataset}")
            
        previous_instructions = shared_memory.instructions_prompt.strip("\n")
        previous_code = shared_memory.codes_prompt.strip("\n")
        execution_results = shared_memory.feedbacks_prompt.strip("\n")
        variables_info = shared_memory.variables_prompt.strip("\n")

        prompt = f"""Task Description
----------------
Review the task description below to understand the problem context:
<TaskDescription>
{task_title}
{task_description}</TaskDescription>

Previous Instructions
---------------------
<PreviousInstructions>
{previous_instructions}
</PreviousInstructions>

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

User Query
----------
This is the query you need to answer:
<Query>{query}</Query>

Instructions
-----------
You need to base on details of the known visual information in the image to answer question. Respond concisely with key terms or names related to the question. 
Base your deductions solely on the Execution Feedback, which provides details of the known visual information in the image. Avoid making any random guesses if the available evidence does not sufficiently support your answer.
For example, When provided with limited information that only identifies an object as a fruit without further details, it's crucial to avoid making arbitrary guesses about the fruit's identity. Instead, the response should acknowledge the insufficiency of the data for a definitive identification.
For VQA tasks, provide a direct natural language answer. If the answer is not in the available variables, respond with "unknown".

Format your response as follows:
<Thinking>Your thinking here</Thinking>
<FinalAnswer>Your concise answer here</FinalAnswer>
"""
        return prompt
