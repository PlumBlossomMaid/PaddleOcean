"""Hook calling utilities - dispatch to callbacks, model, and strategy hooks."""

from typing import Any


def _call_callback_hooks(trainer: Any, hook_name: str, *args: Any, **kwargs: Any) -> None:
    """Call a hook on all callbacks."""
    for cb in getattr(trainer, "callbacks", None) or []:
        fn = getattr(cb, hook_name, None)
        if fn is not None:
            fn(trainer, trainer._model, *args, **kwargs)


def _call_module_hook(trainer: Any, hook_name: str, *args: Any, **kwargs: Any) -> Any:
    """Call a hook on the model."""
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
    """Wrap a trainer entry-point with interrupt handling.

    All entry-point exceptions funnel through here. On an exception we run the
    interrupt path — set INTERRUPTED, dispatch ``on_exception`` hooks,
    finalize loggers as ``"failed"`` and teardown — so resources are released
    and callbacks see the exception before re-raising.
    """
    try:
        return trainer_fn(*args, **kwargs)
    except KeyboardInterrupt as exception:
        _interrupt(trainer, exception)
        trainer._teardown()
        raise
    except BaseException as exception:  # noqa: BLE001 — intentional funnel
        _interrupt(trainer, exception)
        trainer._teardown()
        raise


def _interrupt(trainer: Any, exception: BaseException) -> None:
    """Run the interrupt bookkeeping so a failure still tears down cleanly."""
    from ocean.trainer.states import TrainerStatus

    trainer.state.status = TrainerStatus.INTERRUPTED
    trainer._signal_connector.received_sigterm = True

    # Give callbacks/model a chance to observe the exception before teardown.
    _call_callback_hooks(trainer, "on_exception", exception)
    model = trainer._model
    if model is not None and hasattr(model, "on_exception"):
        model._current_fx_name = "on_exception"
        try:
            model.on_exception(exception)
        finally:
            model._current_fx_name = None

    strategy = getattr(trainer, "strategy", None)
    if strategy is not None and hasattr(strategy, "on_exception"):
        strategy.on_exception(exception)

    trainer._logger_connector.finalize("failed")


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
