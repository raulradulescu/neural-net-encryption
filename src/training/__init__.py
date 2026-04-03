"""ANC training package."""

from importlib import import_module

__all__ = [
    "ANCTrainingConfig",
    "build_models",
    "build_optimizers",
    "evaluate_models",
    "generate_dataset",
    "load_checkpoint",
    "load_config_file",
    "main",
    "save_checkpoint",
    "train_anc",
    "train_batch",
]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(".train_anc", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
