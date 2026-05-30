from __future__ import annotations

from ...llm import llm_with_message
from ...util.config import Config

from ...prompt.answering.answerer_example import answerer_examples

class FinalAnswerer:

    def __init__(self, model_name: str, dataset: str):
        self.model_name = model_name
        self.dataset = dataset
        self.conversation = []
        self.reset()

    def reset(self):
        self.conversation = [
            {"role": "system", "content": "You are an AI assistant designed to assist with compositional visual reasoning tasks by answering questions based on the available visual information."}
        ]

    async def __call__(self, query: str, guess_list: list[str]) -> str | None:
        prompt = self.build_prompt(query, guess_list)

        if Config.debug:
            with open("answerer.txt", "w") as f:
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
    
    def build_prompt(self, query: str, guess_list: list[str]) -> str:
        prompt = f"""
Please answer the following questions using the given guesses in list.
If a unique answer cannot be determined, choose only one of the possible answers.

Examples
--------
<Examples>
{answerer_examples}
</Examples>

User Query
----------
This is the query you need to answer:
<Query>{query}</Query>
<GuessList>{guess_list}</GuessList>


----------
Please give the answer in 1-2 words. Format your response as follows:
<Answer>Your concise answer here</Answer>
"""
        return prompt
