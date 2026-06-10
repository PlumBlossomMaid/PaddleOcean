"""Trainer setup utilities - configuration validation and debug flags."""

from typing import Any, Optional

import paddle


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
    if fast_dev_run:
        n = fast_dev_run if isinstance(fast_dev_run, int) else 1
        trainer.limit_train_batches = n
        trainer.limit_val_batches = n
        trainer.limit_test_batches = n
        trainer.limit_predict_batches = n
        trainer.num_sanity_val_steps = 0


def _verify_loop_configurations(trainer: Any) -> None:
    """Verify that training loop configurations are consistent."""
    if trainer.accumulate_grad_batches < 1:
        raise ValueError(f"accumulate_grad_batches must be >= 1, got {trainer.accumulate_grad_batches}")
