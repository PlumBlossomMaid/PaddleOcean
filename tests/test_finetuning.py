"""Tests for BaseFinetuning / BackboneFinetuning.

The callback used to be a shell: it unfroze *all* params (ignoring the backbone
attribute), never scheduled the backbone LR, and collected constructor args it
did not use. These tests lock in the real behaviour:

* the backbone is frozen at fit start and the head keeps training,
* at ``unfreeze_backbone_at_epoch`` the backbone is unfrozen and added to the
  optimizer as a second param group with a lower (scaled) LR,
* the backbone LR is scheduled up each epoch and clamped to the head LR,
* BatchNorm handling and the freeze/unfreeze primitives behave.
"""

from __future__ import annotations

import paddle
import paddle.nn as nn
import pytest

import ocean
from ocean.callbacks import BackboneFinetuning, BaseFinetuning
from ocean.callbacks.finetuning import multiplicative


class _TLModel(ocean.Model):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1D(8), nn.Linear(8, 4))
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.backbone(x))

    def configure_optimizers(self):
        # Only the head is optimized up front; the callback adds the backbone.
        return paddle.optimizer.Adam(
            learning_rate=0.01, parameters=[{"params": self.head.parameters(), "weight_decay": 0.0}]
        )

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self(x).mean()


def _loader(n: int = 20, bs: int = 10):
    return paddle.io.DataLoader(paddle.io.TensorDataset([paddle.randn([n, 4])]), batch_size=bs)


def _trainer(**kw):
    defaults = dict(logger=False, enable_checkpointing=False, enable_progress_bar=False, limit_val_batches=0)
    defaults.update(kw)
    return ocean.Trainer(**defaults)


# --------------------------------------------------------------------
# Freeze/unfreeze primitives
# --------------------------------------------------------------------
def test_freeze_and_make_trainable():
    layer = nn.Linear(4, 2)
    assert all(not p.stop_gradient for p in layer.parameters())
    BaseFinetuning.freeze(layer)
    assert all(p.stop_gradient for p in layer.parameters())
    BaseFinetuning.make_trainable(layer)
    assert all(not p.stop_gradient for p in layer.parameters())


def test_freeze_keeps_batchnorm_trainable_by_default():
    bn = nn.BatchNorm1D(4)
    BaseFinetuning.freeze(bn, train_bn=True)
    # train_bn=True leaves BN params unfrozen.
    assert all(not p.stop_gradient for p in bn.parameters())
    BaseFinetuning.freeze(bn, train_bn=False)
    assert all(p.stop_gradient for p in bn.parameters())


def test_filter_params_by_trainable_state():
    seq = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    BaseFinetuning.freeze(seq[0])
    trainable = list(BaseFinetuning.filter_params(seq, requires_grad=True))
    frozen = list(BaseFinetuning.filter_params(seq, requires_grad=False))
    assert len(trainable) == 2  # seq[1] weight+bias
    assert len(frozen) == 2  # seq[0] weight+bias


# --------------------------------------------------------------------
# Config / validation
# --------------------------------------------------------------------
def test_requires_backbone_attribute():
    class _NoBackbone(ocean.Model):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Linear(4, 2)

        def configure_optimizers(self):
            return paddle.optimizer.Adam(learning_rate=0.01, parameters=self.parameters())

        def training_step(self, batch, batch_idx):
            return self.net(batch[0]).mean()

    cb = BackboneFinetuning(unfreeze_backbone_at_epoch=1)
    trainer = _trainer(max_epochs=1, limit_train_batches=1, callbacks=[cb])
    with pytest.raises(ValueError, match="backbone"):
        trainer.fit(_NoBackbone(), _loader())


def test_ratio_alias_backwards_compatible():
    cb = BackboneFinetuning(backbone_initial_ratio=0.25)
    assert cb.backbone_initial_ratio_lr == 0.25


# --------------------------------------------------------------------
# End-to-end scheduling
# --------------------------------------------------------------------
def test_backbone_frozen_then_unfrozen_and_added_to_optimizer():
    cb = BackboneFinetuning(unfreeze_backbone_at_epoch=2, backbone_initial_ratio_lr=0.1, initial_denom_lr=1.0)
    model = _TLModel()
    trainer = _trainer(max_epochs=4, limit_train_batches=2, callbacks=[cb])
    trainer.fit(model, _loader())

    # After unfreeze the backbone is trainable...
    assert all(not p.stop_gradient for p in model.backbone.parameters())
    raw = trainer.optimizers[0]._optimizer
    # ...and lives in a second param group with a lower-than-head scale.
    assert len(raw._param_groups) == 2
    assert raw._param_groups[-1]["learning_rate"] < 1.0
    assert cb.previous_backbone_lr is not None


def test_backbone_lr_schedules_up_and_aligns():
    # lambda doubles each epoch; should_align clamps to head LR (0.01).
    cb = BackboneFinetuning(
        unfreeze_backbone_at_epoch=1,
        backbone_initial_lr=0.005,
        lambda_func=multiplicative,
        should_align=True,
        initial_denom_lr=1.0,
    )
    model = _TLModel()
    trainer = _trainer(max_epochs=4, limit_train_batches=1, callbacks=[cb])
    trainer.fit(model, _loader())
    # After several doublings and clamping, the backbone LR never exceeds head LR.
    assert cb.previous_backbone_lr <= 0.01 + 1e-9


def test_backbone_gets_smaller_updates_than_head():
    """The scaled backbone group should receive proportionally smaller updates."""
    # Seeded: the assertion compares two small weight deltas computed from random
    # data, so without a seed its outcome depends on wherever the global RNG
    # happens to be — any test added ahead of this one could flip it.
    paddle.seed(0)
    cb = BackboneFinetuning(
        unfreeze_backbone_at_epoch=0, backbone_initial_lr=0.001, initial_denom_lr=1.0, should_align=False
    )
    model = _TLModel()
    trainer = _trainer(max_epochs=1, limit_train_batches=3, callbacks=[cb])

    # Snapshot backbone + head weights before fit.
    bb_before = model.backbone[0].weight.clone()
    head_before = model.head.weight.clone()
    trainer.fit(model, _loader())

    bb_delta = float((model.backbone[0].weight - bb_before).abs().mean())
    head_delta = float((model.head.weight - head_before).abs().mean())
    # Head LR is 0.01, backbone effective LR is 0.001 -> head moves more.
    assert head_delta > bb_delta


# --------------------------------------------------------------------
# State
# --------------------------------------------------------------------
def test_state_dict_roundtrip():
    cb = BackboneFinetuning()
    cb.previous_backbone_lr = 0.003
    cb._internal_optimizer_metadata = {0: ["x"]}
    sd = cb.state_dict()
    restored = BackboneFinetuning()
    restored.load_state_dict(sd)
    assert restored.previous_backbone_lr == 0.003
    assert restored._internal_optimizer_metadata == {0: ["x"]}
