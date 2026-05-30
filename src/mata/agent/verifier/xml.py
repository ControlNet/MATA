from __future__ import annotations

import re

from .base import Verifier


class XMLTagVerifier(Verifier):
    def __init__(self, tag: str):
        self.tag = tag

    def __call__(self, response: str) -> Verifier.Result:
        pattern = rf"<{self.tag}>(.*?)<\/{self.tag}>"
        try:
            match = re.search(pattern, response, re.DOTALL)
        except TypeError:
            feedback = f"Failed to extract {self.tag}."
            feedback += f"Please provide your {self.tag.lower()} in the format: <{self.tag}>[your {self.tag.lower()} here]</{self.tag}>"
            return self.Failure(feedback=feedback)

        if match:
            return self.Success(match.group(1).strip("\"\n "))
        else:
            feedback = f"Failed to extract {self.tag}."
            feedback += f"Please provide your {self.tag.lower()} in the format: <{self.tag}>[your {self.tag.lower()} here]</{self.tag}>"
            return self.Failure(feedback=feedback)
