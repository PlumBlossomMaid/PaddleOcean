"""Hook calling utilities - dispatch to callbacks, model, and strategy hooks."""

from typing import Any


def _call_callback_hooks(trainer: Any, hook_name: str, *args: Any, **kwargs: Any) -> None:
    """Call a hook on all callbacks."""
    for cb in getattr(trainer, "callbacks", None) or []:
        fn = getattr(cb, hook_name, None)
        if fn is not None:
            fn(trainer, trainer._model, *args, **kwargs)


def _call_lightning_module_hook(trainer: Any, hook_name: str, *args: Any, **kwargs: Any) -> Any:
    """Call a hook on the model/LightningModule."""
    model = trainer._model
    if model is None:
        return None
    fn = getattr(model, hook_name, None)
    if fn is None:
        return None
    # Set the current function name for logging context
    model._current_fx_name = hook_name
    try:
        return fn(*args, **kwargs)
    finally:
        model._current_fx_name = None


def _call_and_handle_interrupt(trainer: Any, trainer_fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Wrap a trainer function with interrupt handling."""
    try:
        return trainer_fn(*args, **kwargs)
    except KeyboardInterrupt:
        trainer._signal_connector.received_sigterm = True
        raise


def _call_setup_hook(trainer: Any) -> None:
    """Call setup hook on model and datamodule."""
    model = trainer._model
    if hasattr(model, "setup"):
        model.setup("fit")
    dm = getattr(trainer, "datamodule", None)
    if dm is not None and hasattr(dm, "setup"):
        dm.setup("fit")


def _call_configure_model(trainer: Any) -> None:
    """Call configure_model hook."""
    model = trainer._model
    if hasattr(model, "configure_model"):
        model.configure_model()
