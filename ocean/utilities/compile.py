"""Compile utilities — paddle.jit.to_static integration for ocean.Model.

Usage:
    model = MyModel()
    model.compile()         # or ocean.utilities.compile.from_compiled(model)
    trainer.fit(model, ...)

    # Revert:
    ocean.utilities.compile.to_uncompiled(model)
"""

from typing import Any, Union

import paddle

import ocean


def from_compiled(model: "ocean.Model") -> "ocean.Model":
    """Apply ``paddle.jit.to_static`` to the model's key methods.

    Wraps forward and step methods with static graph compilation to accelerate training.

    Args:
        model: An ``ocean.Model`` instance.

    Returns:
        The same model instance with compiled methods.
    """
    if not isinstance(model, ocean.Model):
        raise ValueError(f"`model` is required to be an `ocean.Model`. Found `{type(model).__name__}` instead.")

    _compiler_ctx = {
        "compiler": "to_static",
        "original_forward": model.forward,
        "original_training_step": model.training_step,
        "original_validation_step": model.validation_step,
        "original_test_step": model.test_step,
        "original_predict_step": model.predict_step,
    }

    model.forward = paddle.jit.to_static(model.forward)
    model.training_step = paddle.jit.to_static(model.training_step)
    model.validation_step = paddle.jit.to_static(model.validation_step)
    model.test_step = paddle.jit.to_static(model.test_step)
    model.predict_step = paddle.jit.to_static(model.predict_step)

    object.__setattr__(model, "_compiler_ctx", _compiler_ctx)
    return model


def to_uncompiled(model: Union["ocean.Model", Any]) -> "ocean.Model":
    """Reverse compilation, restore original dynamic methods.

    Args:
        model: A compiled ``ocean.Model`` instance.

    Returns:
        The same model instance with original methods restored.
    """
    if isinstance(model, ocean.Model):
        ctx = getattr(model, "_compiler_ctx", None)
        if ctx is None:
            raise ValueError("`model` is not compiled. Found no `_compiler_ctx`.")
        model.forward = ctx["original_forward"]
        model.training_step = ctx["original_training_step"]
        model.validation_step = ctx["original_validation_step"]
        model.test_step = ctx["original_test_step"]
        model.predict_step = ctx["original_predict_step"]
        object.__setattr__(model, "_compiler_ctx", None)
        return model

    raise ValueError(f"`model` must be an instance of `ocean.Model`, got `{type(model).__name__}`")


def _maybe_unwrap_compiled(model: object) -> "ocean.Model":
    """Auto-detect compiled model and return the underlying ``ocean.Model``."""
    if isinstance(model, ocean.Model):
        return model
    raise TypeError(f"`model` must be an `ocean.Model`, got `{type(model).__qualname__}`")


def _verify_strategy_supports_compile(model: "ocean.Model", strategy: Any) -> None:
    """Verify that the current strategy is compatible with compiled models."""
    ctx = getattr(model, "_compiler_ctx", None)
    if ctx is not None:
        supported = ("SingleDeviceStrategy", "DDPStrategy")
        strategy_name = type(strategy).__name__
        if strategy_name not in supported:
            raise RuntimeError(
                f"Compiled model is incompatible with strategy `{strategy_name}`."
                f" Supported strategies: {', '.join(supported)}."
            )
