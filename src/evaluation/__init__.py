"""ANC evaluation package."""

from importlib import import_module

__all__ = [
    "ANCEvalConfig",
    "evaluate_checkpoint",
    "main",
]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(".eval_anc", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
