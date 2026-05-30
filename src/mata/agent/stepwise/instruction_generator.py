from __future__ import annotations

import json
import numpy as np

from ...llm import llm_with_message
from ...memory.shared_memory import SharedMemory
from ...util.config import Config
from ...util.console import logger

from ...prompt.stepwise.instruction import (
    gqa_instruction_setting,
    refcoco_instruction_setting,
    gqa_skills,
    refcoco_skills,
)
from ...prompt.stepwise.instruction_example import (
    gqa_instruction_example,
    refcoco_instruction_example,
)
from ...prompt.stepwise.instruction_task import (
    gqa_task_title,
    refcoco_task_title,
    gqa_task_desc,
    refcoco_task_desc,
)


class InstructionGenerator:

    def __init__(self, model_name: str, dataset: str, num_instructions: int, num_trials: int):
        self.model_name = model_name
        self.dataset = dataset
        self.num_instructions = num_instructions
        self.num_trials = num_trials
        
        self.conversation = []
        self.reset()
    def reset(self):
        self.conversation = [
            {"role": "system", "content": "You are an AI assistant designed to assist with compositional visual reasoning tasks providing valid step by step instruction for answering questions and understanding visual information."}
        ]

    async def __call__(self, query: str, current_step: int, shared_memory: SharedMemory):
        """Generate stepwise reasoning instructions for the current step."""
        prompt = self.build_prompt(query, current_step, shared_memory)
        
        if Config.debug:
            with open("instruction_generator.txt", "w") as f:
                f.write(prompt)

        instructions = []
        probs = []

        # Add the prompt to the conversation
        self.conversation.append({"role": "user", "content": prompt})
        
        for _ in range(self.num_trials):
            if len(instructions) >= self.num_instructions:
                break

            # Get response from the model
            response = await llm_with_message(self.model_name, self.conversation, format="json")
            
            # Save the response to the conversation
            if response is None:
                logger.warning("No response from LLM")
                continue
            
            # Extract the JSON array from the response
            try:
                parsed_instructions = json.loads(response)["instructions"]
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from response")
                continue
            except KeyError:
                logger.warning("Key 'instructions' not found in response")
                continue

            for item in parsed_instructions:
                if 'instruction' in item and 'probability' in item:
                    instructions.append(item['instruction'])
                    probs.append(item['probability'])
                    
                if len(instructions) >= self.num_instructions:
                    break
        
        return instructions, np.array(probs) if len(probs) > 0 else np.array([])

    def build_prompt(self, query: str, current_step: int, shared_memory: SharedMemory):
        match self.dataset:
            case "gqa":
                instruction_setting = gqa_instruction_setting
                instruction_example = gqa_instruction_example
                skills = gqa_skills
                task_title = gqa_task_title
                task_description = gqa_task_desc
            case "refcoco":
                instruction_setting = refcoco_instruction_setting
                instruction_example = refcoco_instruction_example
                skills = refcoco_skills
                task_title = refcoco_task_title
                task_description = refcoco_task_desc
            case _:
                raise ValueError(f"Unsupported dataset: {self.dataset}")
        
        # Extract state information from memory bank
        previous_instructions = shared_memory.instructions_prompt.strip("\n")
        previous_code = shared_memory.codes_prompt.strip("\n")
        execution_results = shared_memory.feedbacks_prompt.strip("\n")
        variables_info = shared_memory.variables_prompt.strip("\n")
        
        # Format the prompt with clear sections
        prompt = f"""Instruction Settings
--------------------
<InstructionSetting>
{instruction_setting}</InstructionSetting>

Skills Overview
---------------
The following are the skills that you can use to solve the query:
<Skills>
{skills}</Skills>

Task Description
----------------
Review the task description below to understand the problem context:
<TaskDescription>
{task_title}
{task_description}</TaskDescription>

Example Instructions
-------------------
How to Use these skills:
<Examples>
{instruction_example}</Examples>

User Query
----------
This is the query you need to solve:
<Query>{query}</Query>

Current Step
------------
<Step>{current_step}</Step>

Previous Instructions
---------------------
<PreviousInstructions>
{previous_instructions}</PreviousInstructions>

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
Based on the current context, generate {self.num_instructions} possible next instructions to help solve the query.
For each instruction, assign a probability score indicating how promising it will lead to the final answer.

Your response must be in this JSON array format:
{{
    "instructions": [
        {{"instruction": "specific instruction", "probability": 0.X}},
        {{"instruction": "another instruction", "probability": 0.Y}},
        ...
    ]
}}
"""
        return prompt
