"""T5: ddp_spawn forwards the full training config to subprocess Trainers.

The spawned Trainer was re-created with only a handful of parameters, silently
dropping gradient clipping, accumulation, batch limits, validation scheduling,
etc. _spawn_trainer_kwargs() now carries them all. (Actual multiprocessing
spawn is not exercised here — only the forwarded configuration.)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ocean


def _trainer():
    return ocean.Trainer(
        max_epochs=7,
        min_epochs=2,
        max_steps=100,
        limit_train_batches=0.5,
        limit_val_batches=3,
        limit_test_batches=4,
        val_check_interval=2,
        check_val_every_n_epoch=2,
        gradient_clip_val=1.5,
        gradient_clip_algorithm="value",
        accumulate_grad_batches=4,
        num_sanity_val_steps=0,
        reload_dataloaders_every_n_epochs=3,
        detect_anomaly=True,
        log_every_n_steps=7,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )


def test_spawn_kwargs_forward_training_config():
    kw = _trainer()._spawn_trainer_kwargs(2)
    expected = {
        "strategy": "ddp",
        "devices": 2,
        "max_epochs": 7,
        "min_epochs": 2,
        "max_steps": 100,
        "limit_train_batches": 0.5,
        "limit_val_batches": 3,
        "limit_test_batches": 4,
        "val_check_interval": 2,
        "check_val_every_n_epoch": 2,
        "gradient_clip_val": 1.5,
        "gradient_clip_algorithm": "value",
        "accumulate_grad_batches": 4,
        "num_sanity_val_steps": 0,
        "reload_dataloaders_every_n_epochs": 3,
        "detect_anomaly": True,
        "log_every_n_steps": 7,
    }
    for k, v in expected.items():
        assert kw[k] == v, f"{k}: {kw.get(k)!r} != {v!r}"


def test_spawned_trainer_has_forwarded_config():
    kw = _trainer()._spawn_trainer_kwargs(2)
    spawned = ocean.Trainer(**kw)
    assert spawned.gradient_clip_val == 1.5
    assert spawned.gradient_clip_algorithm == "value"
    assert spawned.accumulate_grad_batches == 4
    assert spawned.max_steps == 100
    assert spawned.min_epochs == 2
    assert spawned.limit_train_batches == 0.5
    assert spawned.val_check_interval == 2
    assert spawned.check_val_every_n_epoch == 2
    assert spawned.reload_dataloaders_every_n_epochs == 3
    assert spawned.detect_anomaly is True
    assert spawned.log_every_n_steps == 7


def test_spawn_kwargs_reflect_enable_flags_from_callbacks():
    # checkpointing/progress-bar enabled -> reflected in forwarded kwargs
    t = ocean.Trainer(
        max_epochs=1,
        enable_checkpointing=True,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        logger=False,
    )
    kw = t._spawn_trainer_kwargs(2)
    assert kw["enable_checkpointing"] is True
    assert kw["enable_progress_bar"] is True

    t2 = ocean.Trainer(
        max_epochs=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        logger=False,
    )
    kw2 = t2._spawn_trainer_kwargs(2)
    assert kw2["enable_checkpointing"] is False
    assert kw2["enable_progress_bar"] is False
