"""D1 regression: datamodule.prepare_data() is rank-gated and ordered before setup.

Under multi-process training, prepare_data (download/preprocess) must run on a
single process to avoid races on shared storage, and must precede setup().
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ocean.trainer.connectors import _DataConnector


class _FakeStrategy:
    """Minimal strategy exposing rank info and a barrier for the connector."""

    def __init__(self, local_rank=0, node_rank=0):
        self.local_rank = local_rank
        self.node_rank = node_rank
        self.barrier_calls = []

    def barrier(self, name=None):
        self.barrier_calls.append(name)


class _FakeTrainer:
    def __init__(self, strategy):
        self.strategy = strategy
        self.datamodule = None


class _RecordingDataModule:
    def __init__(self, per_node=True):
        self.prepare_data_per_node = per_node
        self.events = []

    def prepare_data(self):
        self.events.append("prepare")

    def setup(self, stage):
        self.events.append(f"setup:{stage}")


def _run_prepare(local_rank, node_rank, per_node=True):
    strat = _FakeStrategy(local_rank=local_rank, node_rank=node_rank)
    trainer = _FakeTrainer(strat)
    dm = _RecordingDataModule(per_node=per_node)
    trainer.datamodule = dm
    conn = _DataConnector(trainer)
    conn.prepare_data()
    return dm, strat


def test_local_rank_zero_prepares():
    dm, strat = _run_prepare(local_rank=0, node_rank=0)
    assert dm.events == ["prepare"]
    assert strat.barrier_calls == ["prepare_data"]


def test_nonzero_local_rank_skips_prepare_but_barriers():
    dm, strat = _run_prepare(local_rank=1, node_rank=0)
    assert dm.events == []  # did NOT download
    assert strat.barrier_calls == ["prepare_data"]  # but waited at the barrier


def test_per_node_true_prepares_on_each_node_local_zero():
    # node 1, local rank 0: with per_node=True this node prepares too
    dm, _ = _run_prepare(local_rank=0, node_rank=1, per_node=True)
    assert dm.events == ["prepare"]


def test_per_node_false_only_global_zero_prepares():
    # node 1, local rank 0: with per_node=False only global rank 0 prepares
    dm, _ = _run_prepare(local_rank=0, node_rank=1, per_node=False)
    assert dm.events == []


def test_no_strategy_defaults_to_prepare():
    """Without a strategy (rank 0 assumed), preparation still runs."""

    class _T:
        strategy = None
        datamodule = None

    t = _T()
    dm = _RecordingDataModule()
    t.datamodule = dm
    _DataConnector(t).prepare_data()
    assert dm.events == ["prepare"]


def test_fit_prepares_before_setup_and_before_reading_the_data():
    """The fit path must run prepare_data, then setup('fit'), then read the
    dataloaders — each step depends on the one before it.

    prepare_data and setup are dispatched by the Trainer (setup also has to
    reach the callbacks and the model); attach_data only reads what they built.
    """
    import paddle

    import ocean

    events = []

    class _DM(ocean.DataModule):
        def prepare_data(self):
            events.append("prepare")

        def setup(self, stage):
            events.append(f"setup:{stage}")
            self.dataset = paddle.io.TensorDataset([paddle.randn([8, 4]), paddle.randint(0, 2, [8])])

        def train_dataloader(self):
            events.append("train_dataloader")
            return paddle.io.DataLoader(self.dataset, batch_size=4)

        def val_dataloader(self):
            return paddle.io.DataLoader(self.dataset, batch_size=4)

    class _M(ocean.Model):
        def __init__(self):
            super().__init__()
            self.linear = paddle.nn.Linear(4, 2)

        def forward(self, x):
            return self.linear(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            return paddle.nn.functional.cross_entropy(self(x), y)

        def configure_optimizers(self):
            return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())

    trainer = ocean.Trainer(
        max_epochs=1, verbose=0, logger=False, enable_checkpointing=False, enable_progress_bar=False
    )
    trainer.fit(_M(), datamodule=_DM())

    assert events[:3] == ["prepare", "setup:fit", "train_dataloader"]
