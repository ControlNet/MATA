from importlib import import_module, util

from ._base import BaseModel, module_registry
from ..util.console import logger

__all__ = ["BaseModel", "module_registry"]

_florence = import_module(".florence", __name__)
_internvl = import_module(".internvl", __name__)
_depth_anything = import_module(".depth_anything", __name__)
_xvlm = import_module(".xvlm", __name__)

if util.find_spec("groundingdino") is None:
    logger.info("GroundingDino not installed. Skipping.")
else:
    _grounding_dino = import_module(".grounding_dino", __name__)
