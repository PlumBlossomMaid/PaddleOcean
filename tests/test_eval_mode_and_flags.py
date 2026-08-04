"""Tests for eval-mode hooks and Trainer flag handling.

Covers:
- validate()/test()/sanity-check switch modes through the model hooks and
  restore the exact per-sublayer modes afterwards
- num_sanity_val_steps=-1 sanity checks the whole validation set
- barebones=True rejects flags it would otherwise silently override
- the tuner skips its finders under fast_dev_run
"""

import paddle
import pytest

import ocean
from ocean.model import Model

# ── Model / loaders ──────────────────────────────────────────────────────────


class HookModel(Model):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 2)
        self.dropout = paddle.nn.Dropout(0.5)
        self.hooks = []
        self.val_batches = []
        self.test_batches = []

    def forward(self, x):
        return self.linear(self.dropout(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        return paddle.nn.functional.cross_entropy(self(x), y)

    def validation_step(self, batch, batch_idx):
        self.val_batches.append((batch_idx, self.training))

    def test_step(self, batch, batch_idx):
        self.test_batches.append((batch_idx, self.training))

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())

    def on_validation_model_eval(self):
        self.hooks.append("val_model_eval")
        super().on_validation_model_eval()

    def on_test_model_eval(self):
        self.hooks.append("test_model_eval")
        super().on_test_model_eval()


def make_loader(n=32, bs=8):
    ds = paddle.io.TensorDataset([paddle.randn([n, 10]), paddle.randint(0, 2, [n])])
    return paddle.io.DataLoader(ds, batch_size=bs)


def make_trainer(**kwargs):
    kwargs.setdefault("max_epochs", 1)
    kwargs.setdefault("verbose", 0)
    kwargs.setdefault("logger", False)
    kwargs.setdefault("enable_checkpointing", False)
    kwargs.setdefault("enable_progress_bar", False)
    return ocean.Trainer(**kwargs)


# ── Eval-mode hooks ──────────────────────────────────────────────────────────


def test_validate_uses_the_model_eval_hook():
    model = HookModel()
    make_trainer().validate(model, dataloaders=make_loader())
    assert model.hooks == ["val_model_eval"]
    assert all(training is False for _, training in model.val_batches)


def test_test_uses_the_model_eval_hook():
    model = HookModel()
    make_trainer().test(model, dataloaders=make_loader())
    assert model.hooks == ["test_model_eval"]
    assert all(training is False for _, training in model.test_batches)


def test_sanity_check_uses_the_model_eval_hook():
    model = HookModel()
    make_trainer(num_sanity_val_steps=1).fit(model, train_dataloaders=make_loader(), val_dataloaders=make_loader())
    assert "val_model_eval" in model.hooks


@pytest.mark.parametrize("entry_point", ["validate", "test"])
def test_evaluation_restores_previous_module_modes(entry_point):
    """A sublayer deliberately kept in eval mode must not be switched back on."""
    model = HookModel()
    model.train()
    model.dropout.eval()

    getattr(make_trainer(), entry_point)(model, dataloaders=make_loader())

    assert model.training is True
    assert model.linear.training is True
    assert model.dropout.training is False


def test_evaluation_does_not_force_train_mode_on_an_eval_model():
    """A model handed to validate() in eval mode stays in eval mode."""
    model = HookModel()
    model.eval()
    make_trainer().validate(model, dataloaders=make_loader())
    assert model.training is False


# ── num_sanity_val_steps ─────────────────────────────────────────────────────


def test_num_sanity_val_steps_minus_one_runs_the_whole_val_set():
    model = HookModel()
    trainer = make_trainer(num_sanity_val_steps=-1, limit_train_batches=1)
    assert trainer.num_sanity_val_steps == float("inf")

    trainer.fit(model, train_dataloaders=make_loader(), val_dataloaders=make_loader(32))
    # 4 sanity batches, then the epoch-end validation.
    assert len(model.val_batches) >= 4


def test_num_sanity_val_steps_default_and_explicit():
    assert make_trainer().num_sanity_val_steps == 2
    assert make_trainer(num_sanity_val_steps=0).num_sanity_val_steps == 0
    assert make_trainer(num_sanity_val_steps=5).num_sanity_val_steps == 5


# ── barebones ────────────────────────────────────────────────────────────────


def test_barebones_disables_everything_by_default():
    trainer = ocean.Trainer(barebones=True, verbose=0)
    assert trainer.loggers == []
    assert trainer.log_every_n_steps == 0
    assert trainer.num_sanity_val_steps == 0
    assert trainer.enable_model_summary is False
    assert not any(cb.__class__.__name__ == "ModelCheckpoint" for cb in trainer.callbacks)


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("enable_checkpointing", True),
        ("enable_progress_bar", True),
        ("enable_model_summary", True),
        ("logger", True),
        ("log_every_n_steps", 10),
        ("num_sanity_val_steps", 2),
        ("fast_dev_run", True),
        ("detect_anomaly", True),
    ],
)
def test_barebones_rejects_conflicting_flags(flag, value):
    with pytest.raises(ValueError, match="barebones=True"):
        ocean.Trainer(barebones=True, verbose=0, **{flag: value})


def test_barebones_accepts_explicitly_disabled_flags():
    ocean.Trainer(barebones=True, verbose=0, enable_checkpointing=False, num_sanity_val_steps=0, logger=False)


# ── Tuner under fast_dev_run ─────────────────────────────────────────────────


def test_tuner_skips_under_fast_dev_run(capsys):
    model = HookModel()
    trainer = make_trainer(fast_dev_run=True)

    assert trainer.scale_batch_size(model, make_loader()) is None
    assert trainer.lr_find(model, make_loader()) is None

    out = capsys.readouterr().out
    assert "Skipping batch size scaler" in out
    assert "Skipping learning rate finder" in out
