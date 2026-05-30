from __future__ import annotations

import re

from .base import Verifier


class CodeVerifier(Verifier):
    def __call__(self, response: str) -> Verifier.Result:
        # Pattern 1: <PythonCode>```python ... ```</PythonCode>
        pattern1 = r"<PythonCode>\s*\`\`\`python\s*(.*?)\s*\`\`\`\s*<\/PythonCode>"
        # Pattern 2: ```python<PythonCode> ... </PythonCode>```
        pattern2 = r"\`\`\`python\s*<PythonCode>\s*(.*?)\s*<\/PythonCode>\s*\`\`\`"
        # Pattern 3: <PythonCode> ... </PythonCode>
        pattern3 = r"<PythonCode>\s*(.*?)\s*</PythonCode>"
        # Pattern 4: ```python ... ```
        pattern4 = r"\`\`\`python\s*(.*?)\s*\`\`\`"

        # Try each pattern in order
        for pattern in [pattern1, pattern2, pattern3, pattern4]:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                code = match.group(1).strip()
                break
        else:
            # If none of the patterns match
            feedback = "Failed to extract code. Please provide your code in the formats:\n"
            feedback += "<PythonCode>```python\n[your code here]\n```</PythonCode>"
            return self.Failure(feedback)

        # If the code contains "input("
        if "input(" in code:
            feedback = "Do not use input() in your code."
            return self.Failure(feedback)

        return self.Success(code)
