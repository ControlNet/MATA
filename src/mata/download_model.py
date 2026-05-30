import argparse
from pathlib import Path

from dotenv import load_dotenv

from .util.config import Config
from .util.console import console


def required_model_names() -> list[str]:
    model_names = [
        Config.base_config["depth_model"],
        Config.base_config["grounding_model"],
        Config.base_config["vlm_model"],
        Config.base_config["vlm_caption_model"],
        Config.base_config["verify_property_model"],
    ]
    return list(dict.fromkeys(model_names))


def prepare_models(extra_packages: list[str]):
    for package in extra_packages:
        __import__(package)

    from .tool import module_registry

    for model_name in required_model_names():
        ModelClass = module_registry[model_name]
        ModelClass.prepare()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, required=True)
    parser.add_argument("--extra_packages", type=str, nargs="*", default=[])
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    Config.base_config_path = args.base_config
    Config.debug = False

    with console.status("[bold green]Downloading models..."):
        prepare_models(args.extra_packages)


if __name__ == '__main__':
    main()
