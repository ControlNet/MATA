import asyncio
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from pathlib import Path
from typing import Any

import argparse
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--base_config", type=str, required=True)
parser.add_argument("--result_folder", type=str, default="./result")
parser.add_argument("--max_samples", type=int, default=1000)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from mata.util.config import Config
# Load configs
Config.base_config_path = args.base_config
if args.debug:
    Config.base_config["debug"] = True

from mata.util.console import logger
from mata.execution.toolbox import Toolbox
from mata.llm import Cost

from mata.agent.mata import MATA
from eval_data import exp_datasets


def _hyper_agent_run_name() -> str:
    model_name = Config.base_config["hyper_agent"]["model_name"]
    return str(model_name).rstrip("/").replace("/", "__")


def _read_completed_ids(save_path: str) -> set[Any]:
    completed: set[Any] = set()
    if not os.path.exists(save_path):
        return completed
    try:
        with open(save_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    datum_id = obj.get("datum_id")
                    if datum_id is not None:
                        completed.add(datum_id)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return completed


async def main():
    # Initialize tools used by MATA agents.
    Config.model_config = {"cuda": {0: MATA.required_models()}}
    Toolbox.init(["mata.tool"])

    # Select dataset
    match Config.base_config["dataset"]:
        case "gqa":
            dataset = exp_datasets.GQA(args.data_root)
        case "refcoco":
            dataset = exp_datasets.Refcoco(args.data_root, split="testA")
        case _:
            raise ValueError("Only gqa and refcoco are supported in release configs")
        
    hyper_agent_name = _hyper_agent_run_name()

    # Prepare results path and resume list
    Path(args.result_folder).mkdir(parents=True, exist_ok=True)
    save_path = str(Path(args.result_folder) / f"result.mata_{Config.base_config['dataset']}.{hyper_agent_name}.jsonl")
    logger.info(f"Saving results to {save_path}")
    completed = _read_completed_ids(save_path)

    import time
    start_time = time.time()

    # Iterate
    for i in range(min(len(dataset), args.max_samples)):
        sample = dataset[i]
        if sample is None:
            continue
        image_path, datum_id, query, ground_truth = sample

        if datum_id in completed:
            logger.info(f"Skipping {i+1}/{len(dataset)}")
            continue

        logger.info(f"Processing {i+1}/{len(dataset)} | {query}")

        mata = MATA(image_path=image_path, query=query)

        # Drive the state machine until completion
        final_answer = None
        while True:
            result = await mata.step()
            if isinstance(result, MATA.Success):
                final_answer = result.final_answer
                break
            if isinstance(result, MATA.Failure):
                final_answer = None
                break

        # Normalize result for saving
        result_payload: Any
        if final_answer is None:
            result_payload = None
        elif Config.base_config.get("task") == "grounding":
            # Convert list[ImagePatch] -> list[list[float]] bboxes
            if isinstance(final_answer, list):
                result_payload = [patch.to_bbox()[:4] for patch in final_answer]
            else:
                result_payload = []
        else:
            # VQA
            result_payload = final_answer

        with open(save_path, "a") as f:
            f.write(json.dumps({
                "datum_id": datum_id,
                "query": query,
                "ground_truth": ground_truth,
                "result": result_payload
            }) + "\n")

        logger.info(f"Cost: {Cost.cost:.5f}, Input Tokens: {Cost.input_tokens}, Output Tokens: {Cost.output_tokens}")
        logger.info(f"Time: {time.time() - start_time:.5f}s")


if __name__ == "__main__":
    asyncio.run(main())
