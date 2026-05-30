import logging
from typing import TYPE_CHECKING

logging.getLogger("aiosqlite").setLevel(logging.WARN)

if TYPE_CHECKING:
    from .agent.mata import MATA

__all__ = ["MATA"]


def __getattr__(name: str) -> object:
    if name == "MATA":
        from .agent.mata import MATA

        return MATA
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
