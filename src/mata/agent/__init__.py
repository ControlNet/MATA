__all__ = ["MATA", "HyperAgent", "AutoHyperAgent"]


def __getattr__(name: str):
    if name == "MATA":
        from .mata import MATA

        return MATA
    if name in {"HyperAgent", "AutoHyperAgent"}:
        from .hyper_agent import AutoHyperAgent, HyperAgent

        return {"HyperAgent": HyperAgent, "AutoHyperAgent": AutoHyperAgent}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
