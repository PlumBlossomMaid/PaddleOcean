"""Tests for how `Trainer(logger=...)` is resolved.

Covers:
- a logger that implements the interface without subclassing Logger is used
- bool / None / list forms still resolve as before
"""

import paddle

import ocean
from ocean.loggers import CSVLogger
from ocean.loggers.base import Logger


class DuckLogger:
    """Implements the logger interface without inheriting from Logger."""

    def __init__(self):
        self.rows = []

    def log_metrics(self, metrics, step=None):
        self.rows.append(dict(metrics))


class SubclassLogger(Logger):
    def __init__(self):
        self.rows = []

    @property
    def name(self):
        return "spy"

    @property
    def version(self):
        return "0"

    def log_metrics(self, metrics, step=None):
        self.rows.append(dict(metrics))


class TinyModel(ocean.Model):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.log("some_metric", 7.0)
        return paddle.nn.functional.cross_entropy(self(x), y)

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.1, parameters=self.parameters())


def make_loader(n=16, bs=8):
    ds = paddle.io.TensorDataset([paddle.randn([n, 4]), paddle.randint(0, 2, [n])])
    return paddle.io.DataLoader(ds, batch_size=bs)


def make_trainer(**kwargs):
    kwargs.setdefault("verbose", 0)
    kwargs.setdefault("enable_checkpointing", False)
    kwargs.setdefault("enable_progress_bar", False)
    kwargs.setdefault("enable_model_summary", False)
    return ocean.Trainer(**kwargs)


# ── Duck-typed loggers ───────────────────────────────────────────────────────


def test_a_duck_typed_logger_is_used():
    """It used to fall through every branch, leaving trainer.loggers empty —
    the run trained normally with every metric going nowhere."""
    logger = DuckLogger()
    trainer = make_trainer(max_epochs=1, log_every_n_steps=1, logger=logger)
    trainer.fit(TinyModel(), train_dataloaders=make_loader())

    assert trainer.loggers == [logger]
    assert logger.rows and "some_metric" in logger.rows[0]


def test_a_subclass_logger_still_works():
    logger = SubclassLogger()
    trainer = make_trainer(max_epochs=1, log_every_n_steps=1, logger=logger)
    trainer.fit(TinyModel(), train_dataloaders=make_loader())
    assert logger.rows


def test_a_list_of_duck_typed_loggers():
    loggers = [DuckLogger(), DuckLogger()]
    trainer = make_trainer(max_epochs=1, log_every_n_steps=1, logger=loggers)
    trainer.fit(TinyModel(), train_dataloaders=make_loader())
    assert all(lg.rows for lg in loggers)


# ── The existing forms ───────────────────────────────────────────────────────


def test_false_means_no_logger():
    assert make_trainer(logger=False).loggers == []


def test_none_and_true_give_the_default_csv_logger():
    assert isinstance(make_trainer().loggers[0], CSVLogger)
    assert isinstance(make_trainer(logger=True).loggers[0], CSVLogger)


def test_none_entries_in_a_list_are_dropped():
    logger = DuckLogger()
    assert make_trainer(logger=[logger, None]).loggers == [logger]
