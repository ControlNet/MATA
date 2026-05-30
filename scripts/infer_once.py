import argparse
import sys
import asyncio
from typing import Any
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--base_config", type=str, required=True, help="Base config path.")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--image", type=str, default="example2.png", help="Image path to process.")
parser.add_argument(
    "--query",
    type=str,
    default="How many red cups on the left table? Directly give the answer, don't explain.",
    help="Question/query about the image.",
)
args = parser.parse_args()

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from mata.util.config import Config

Config.base_config_path = args.base_config
if args.debug:
    Config.base_config["debug"] = True

from mata.util.console import logger


async def run_mata(image_path: str, query: str) -> Any:
    from mata.execution.toolbox import Toolbox
    from mata.agent.mata import MATA

    Config.model_config = {"cuda": {0: MATA.required_models()}}
    Toolbox.init(["mata.tool"])

    mata = MATA(image_path=image_path, query=query)

    final_answer = None
    while True:
        result = await mata.step()
        if isinstance(result, MATA.Success):
            final_answer = result.final_answer
            break
        if isinstance(result, MATA.Failure):
            final_answer = None
            break

    if final_answer is None:
        return None
    if Config.base_config.get("task") == "grounding":
        try:
            return [patch.to_bbox()[:4] for patch in final_answer]  # type: ignore[attr-defined]
        except Exception:
            return []
    return final_answer


async def main():
    logger.info(f"Processing query: {args.query} | mode=mata")
    result_payload = await run_mata(args.image, args.query)
    logger.info(f"Query: {args.query} Answer: {result_payload}")


if __name__ == "__main__":
    asyncio.run(main())
