from __future__ import annotations
from dataclasses import dataclass

from ...util.config import Config
from ...memory.shared_memory import SharedMemory

from .summarizer import Summarizer
from .final_answerer import FinalAnswerer
from ..verifier import XMLTagVerifier
from ...execution.image_patch import ImagePatch
from ...prompt.stepwise.code_task import (
    gqa_task_desc,
    refcoco_task_desc,
)
from ..stepwise.agent import StepwiseReasoner
from ..oneshot.agent import OneshotReasoner

class Answerer:

    @dataclass
    class Completed:
        answer: str | list[ImagePatch]

    @dataclass
    class Incomplete:
        pass

    @dataclass
    class Failure:
        feedback: str

    Result = Completed | Incomplete | Failure

    def __init__(self, model_name: str, dataset: str, query: str, shared_memory: SharedMemory) -> None:
        self.model_name = model_name
        self.dataset = dataset
        self.query = query
        self.task = Config.base_config["task"]
        self.guesses = []
        match dataset:
            case "gqa":
                self.task_desc = gqa_task_desc
            case "refcoco":
                self.task_desc = refcoco_task_desc
            case _:
                raise ValueError(f"Dataset {dataset} is not supported")
        
        # Summarizer
        self.num_summarizer_trials = Config.base_config["answerer"]["num_summarizer_trials"]
        self.summarizer = Summarizer(self.model_name, self.dataset)
        self.summarization_verifier = XMLTagVerifier("FinalAnswer")

        # Answerer
        self.num_answerer_trials = Config.base_config["answerer"]["num_answerer_trials"]
        self.final_answerer = FinalAnswerer(self.model_name, self.dataset)
        self.answer_verifier = XMLTagVerifier("Answer")

        self.shared_memory = shared_memory

    async def __call__(self, reasoner_return: StepwiseReasoner.Result | OneshotReasoner.Result | None = None) -> Result:
        # ----------------- Summarizer -----------------
        if self.task == "vqa":
            self.summarizer.reset()
            feedback = ""
            first_attempt_summarizer = True
            summarizer_success = False
            for _ in range(self.num_summarizer_trials):
                if first_attempt_summarizer:
                    summarizer_return = await self.summarizer(self.query, self.shared_memory)
                    if summarizer_return is not None:
                        first_attempt_summarizer = False
                else:
                    summarizer_return = await self.summarizer.feedback(feedback)
                    if summarizer_return is None:
                        feedback = "Failed to generate summary."
                        continue
                
                summarization_verifier_return = self.summarization_verifier(summarizer_return)
                match summarization_verifier_return:
                    case XMLTagVerifier.Success("unknown"):
                        summarizer_success = True
                        break
                    case XMLTagVerifier.Success(answer):
                        self.guesses.append(answer)
                        summarizer_success = True
                        break
                    case XMLTagVerifier.Failure(fb):
                        feedback = fb
                        continue
                    case _:
                        raise ValueError(f"Invalid summarization verifier return: {summarization_verifier_return}")
            
            if not summarizer_success:
                return self.Failure(feedback)

        match reasoner_return:
            case StepwiseReasoner.Complete(result) | OneshotReasoner.Success(result):
                if self.task == "vqa":
                    feedback = ""
                    first_attempt_answerer = True
                    answerer_success = False

                    for _ in range(self.num_answerer_trials):
                        if first_attempt_answerer:
                            final_answer = await self.final_answerer(self.query, self.guesses)
                            if final_answer is not None:
                                first_attempt_answerer = False
                        else:
                            final_answer = await self.final_answerer.feedback(feedback)
                            if final_answer is None:
                                feedback = "Failed to generate final answer."
                                continue
                            
                        answer_verifier_return = self.answer_verifier(final_answer)
                        match answer_verifier_return:
                            case XMLTagVerifier.Success(answer):
                                final_answer = answer
                                answerer_success = True
                                break
                            case XMLTagVerifier.Failure(fb):
                                feedback = fb
                                continue
                            case _:
                                raise ValueError(f"Invalid answer verifier return: {answer_verifier_return}")
                    
                    if not answerer_success:
                        return self.Failure(feedback)
                    
                    assert final_answer is not None, "Final answer is None"
                else:
                    final_answer = result
                return self.Completed(final_answer)
            case StepwiseReasoner.Incomplete():
                return self.Incomplete()
            case StepwiseReasoner.Failure(feedback) | OneshotReasoner.Failure(feedback):
                return self.Failure(feedback)
            case _:
                raise ValueError(f"Invalid reasoner return: {reasoner_return}")
