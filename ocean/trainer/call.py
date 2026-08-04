"""Hook calling utilities - dispatch to callbacks, model, and strategy hooks."""

from typing import Any


def _get_profiler(trainer: Any) -> Any:
    """Return the trainer's profiler, or a no-op fallback.

    Callers pass an ``Any`` trainer, so a profiler is not guaranteed to be
    present (e.g. lightweight test doubles); fall back to a PassThroughProfiler
    so the ``profile`` context manager is always available.
    """
    profiler = getattr(trainer, "profiler", None)
    if profiler is None:
        from ocean.profilers import PassThroughProfiler

        profiler = PassThroughProfiler()
    return profiler


def _call_callback_hooks(trainer: Any, hook_name: str, *args: Any, **kwargs: Any) -> None:
    """Call a hook on all callbacks."""
    profiler = _get_profiler(trainer)
    for cb in getattr(trainer, "callbacks", None) or []:
        fn = getattr(cb, hook_name, None)
        if fn is not None:
            state_key = getattr(cb, "state_key", cb.__class__.__name__)
            with profiler.profile(f"[Callback]{state_key}.{hook_name}"):
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
    profiler = _get_profiler(trainer)
    try:
        with profiler.profile(f"[Model]{model.__class__.__name__}.{hook_name}"):
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


def _call_setup_hook(trainer: Any, stage: str = "fit") -> None:
    """Call the ``setup`` hook on the datamodule, the callbacks and the model.

    ``stage`` is the entry point being run (``fit``/``validate``/``test``/
    ``predict``), not always ``"fit"``: a datamodule that builds different
    datasets per stage has to be told which one is starting.
    """
    dm = getattr(trainer, "datamodule", None)
    if dm is not None and hasattr(dm, "setup"):
        dm.setup(stage)
    _call_callback_hooks(trainer, "setup", stage=stage)
    model = trainer._model
    if hasattr(model, "setup"):
        model.setup(stage)


def _call_teardown_hook(trainer: Any, stage: str = "fit") -> None:
    """Call the ``teardown`` hook on the datamodule, the callbacks and the model.

    The mirror image of ``_call_setup_hook``: anything opened in ``setup`` gets
    a chance to close, whichever entry point ran.
    """
    dm = getattr(trainer, "datamodule", None)
    if dm is not None and hasattr(dm, "teardown"):
        dm.teardown(stage)
    _call_callback_hooks(trainer, "teardown", stage=stage)
    model = getattr(trainer, "_model", None)
    if model is not None and hasattr(model, "teardown"):
        model.teardown(stage)
    if model is not None:
        model._current_fx_name = None


def _call_configure_model(trainer: Any) -> None:
    """Call configure_model hook."""
    model = trainer._model
    if hasattr(model, "configure_model"):
        model.configure_model()
