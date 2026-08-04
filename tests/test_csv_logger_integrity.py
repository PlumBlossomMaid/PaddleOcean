"""Tests for the CSV logger's file integrity.

Covers:
- a metric that appears later widens the header instead of corrupting the file
- column order is the same on every run
- hyperparameters are written, not dropped
"""

import csv
import os
import subprocess
import sys
import tempfile

import paddle
import pytest

import ocean
from ocean.loggers import CSVLogger


def read_rows(logger):
    with open(os.path.join(logger.log_dir, "metrics.csv"), newline="") as f:
        return list(csv.DictReader(f))


def header_of(logger):
    with open(os.path.join(logger.log_dir, "metrics.csv")) as f:
        return f.readline().strip()


# ── File stays a valid CSV ───────────────────────────────────────────────────


def test_a_late_metric_rewrites_the_header(tmp_path):
    """Validation metrics first appear partway through a run. Appending them
    under the old header wrote rows with more fields than it declared."""
    logger = CSVLogger(root_dir=str(tmp_path))
    logger.log_metrics({"loss": 1.0}, step=0)
    logger.save()
    logger.log_metrics({"loss": 0.5, "val_acc": 0.9}, step=1)
    logger.save()

    assert header_of(logger) == "step,loss,val_acc"
    assert read_rows(logger) == [
        {"step": "0", "loss": "1.0", "val_acc": ""},
        {"step": "1", "loss": "0.5", "val_acc": "0.9"},
    ]


def test_every_row_matches_the_header(tmp_path):
    logger = CSVLogger(root_dir=str(tmp_path))
    for step, metrics in enumerate([{"a": 1}, {"a": 2, "b": 3}, {"a": 4, "b": 5, "c": 6}]):
        logger.log_metrics(metrics, step=step)
        logger.save()

    with open(os.path.join(logger.log_dir, "metrics.csv"), newline="") as f:
        rows = list(csv.reader(f))
    width = len(rows[0])
    assert all(len(row) == width for row in rows)
    assert width == 4  # step + a + b + c


def test_earlier_rows_keep_their_values(tmp_path):
    logger = CSVLogger(root_dir=str(tmp_path))
    logger.log_metrics({"loss": 1.5}, step=0)
    logger.save()
    logger.log_metrics({"loss": 0.5, "acc": 0.9}, step=1)
    logger.save()

    assert read_rows(logger)[0]["loss"] == "1.5"


# ── Deterministic column order ───────────────────────────────────────────────


def test_column_order_is_sorted(tmp_path):
    logger = CSVLogger(root_dir=str(tmp_path))
    logger.log_metrics({"ff": 6, "aa": 1, "cc": 3, "bb": 2}, step=0)
    logger.save()
    assert header_of(logger) == "step,aa,bb,cc,ff"


def test_column_order_is_stable_across_processes():
    """String hashing is randomised per process, so iterating a set for the
    header gave a different order every run."""
    script = (
        "import os, tempfile;"
        "from ocean.loggers import CSVLogger;"
        "d = tempfile.mkdtemp();"
        "lg = CSVLogger(root_dir=d);"
        "lg.log_metrics({'ff':6,'aa':1,'cc':3,'bb':2,'dd':4,'ee':5}, step=0);"
        "lg.save();"
        "print(open(os.path.join(lg.log_dir,'metrics.csv')).readline().strip())"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    headers = {
        subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env).stdout.strip()
        for _ in range(3)
    }
    assert headers == {"step,aa,bb,cc,dd,ee,ff"}


# ── Hyperparameters ──────────────────────────────────────────────────────────


def test_hyperparameters_are_written(tmp_path):
    """The base class treats log_hyperparams as a no-op, so the default logger
    dropped everything handed to it."""
    yaml = pytest.importorskip("yaml", reason="hyperparameters are written as YAML")
    logger = CSVLogger(root_dir=str(tmp_path))
    logger.log_hyperparams({"lr": 0.1, "batch_size": 8})

    path = os.path.join(logger.log_dir, "hparams.yaml")
    assert os.path.exists(path)
    with open(path) as f:
        assert yaml.safe_load(f) == {"lr": 0.1, "batch_size": 8}


def test_missing_pyyaml_is_reported(tmp_path, monkeypatch, capsys):
    """Without PyYAML the file cannot be written; that has to be said, not
    silently skipped."""
    import builtins

    real_import = builtins.__import__

    def no_yaml(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("no yaml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_yaml)
    CSVLogger(root_dir=str(tmp_path)).log_hyperparams({"lr": 0.1})

    assert "PyYAML is not installed" in capsys.readouterr().out


def test_empty_hyperparameters_write_nothing(tmp_path):
    logger = CSVLogger(root_dir=str(tmp_path))
    logger.log_hyperparams({})
    assert not os.path.exists(os.path.join(logger.log_dir, "hparams.yaml"))


# ── Through a real run ───────────────────────────────────────────────────────


def test_a_training_run_produces_a_parseable_file():
    class Model(ocean.Model):
        def __init__(self):
            super().__init__()
            self.linear = paddle.nn.Linear(4, 2)

        def forward(self, x):
            return self.linear(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            loss = paddle.nn.functional.cross_entropy(self(x), y)
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            self.log("val_loss", paddle.nn.functional.cross_entropy(self(x), y))

        def configure_optimizers(self):
            return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())

    def loader(n=16):
        ds = paddle.io.TensorDataset([paddle.randn([n, 4]), paddle.randint(0, 2, [n])])
        return paddle.io.DataLoader(ds, batch_size=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = CSVLogger(root_dir=tmpdir)
        trainer = ocean.Trainer(
            max_epochs=2,
            log_every_n_steps=1,
            logger=logger,
            verbose=0,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(Model(), train_dataloaders=loader(), val_dataloaders=loader())

        with open(os.path.join(logger.log_dir, "metrics.csv"), newline="") as f:
            rows = list(csv.reader(f))
        assert all(len(row) == len(rows[0]) for row in rows)
