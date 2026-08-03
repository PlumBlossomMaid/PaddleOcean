"""Trainer setup utilities - configuration validation and debug flags."""

from typing import Any

from ocean.utils import MisconfigurationException
from ocean.utils.rank_zero import rank_zero_info


def _init_debugging_flags(
    trainer: Any,
    limit_train_batches: Any,
    limit_val_batches: Any,
    limit_test_batches: Any,
    limit_predict_batches: Any,
    overfit_batches: Any,
    val_check_interval: Any,
    fast_dev_run: Any,
    accumulate_grad_batches: int,
    detect_anomaly: bool,
) -> None:
    """Initialize debugging/training flags based on fast_dev_run/overfit."""
    if not isinstance(fast_dev_run, (bool, int)) or (isinstance(fast_dev_run, int) and fast_dev_run < 0):
        raise MisconfigurationException(
            f"fast_dev_run={fast_dev_run!r} is not a valid configuration. It should be >= 0."
        )

    # Normalize ``fast_dev_run=1`` to ``True`` so downstream ``bool(...)`` checks
    # and logging read the same for the one-batch case.
    trainer.fast_dev_run = True if fast_dev_run == 1 else fast_dev_run

    if fast_dev_run:
        num_batches = int(fast_dev_run)
        trainer.limit_train_batches = num_batches
        trainer.limit_val_batches = num_batches
        trainer.limit_test_batches = num_batches
        trainer.limit_predict_batches = num_batches
        # Bound the run itself, not just the per-epoch batch count: without these
        # the loop keeps replaying ``num_batches`` batches for the full default
        # ``max_epochs``, which is the opposite of a fast debug run.
        trainer.max_epochs = 1
        trainer.max_steps = num_batches
        trainer.num_sanity_val_steps = 0
        trainer.val_check_interval = 1.0
        trainer.check_val_every_n_epoch = 1
        # A time-based validation interval is meaningless for a fixed, tiny
        # number of batches.
        trainer._val_check_time_interval = None
        rank_zero_info(
            f"Running in `fast_dev_run` mode: will run the requested loop using {num_batches} batch(es). "
            "Logging and checkpointing is suppressed."
        )

    if overfit_batches:
        if isinstance(overfit_batches, (int, float)):
            trainer.limit_train_batches = overfit_batches
            trainer.limit_val_batches = overfit_batches
        trainer.overfit_batches = overfit_batches


def _verify_loop_configurations(trainer: Any) -> None:
    """Verify that training loop configurations are consistent."""
    if trainer.accumulate_grad_batches < 1:
        raise ValueError(f"accumulate_grad_batches must be >= 1, got {trainer.accumulate_grad_batches}")
