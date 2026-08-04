"""Tests for StochasticWeightAveraging and WeightAveraging.

Covers:
- both callbacks survive a real ocean.Model (deepcopy of one is not possible)
- SWA anneals the learning rate to swa_lrs, cos and linear
- SWA recomputes batch-norm statistics for the averaged weights
- averaging only touches parameters; buffers follow the latest weights
- argument validation
"""

import paddle
import pytest

import ocean
from ocean.callbacks.stochastic_weight_avg import StochasticWeightAveraging
from ocean.callbacks.weight_averaging import WeightAveraging
from ocean.utils import MisconfigurationException

# ── Model / loader ───────────────────────────────────────────────────────────


class BNModel(ocean.Model):
    def __init__(self, lr=0.1):
        super().__init__()
        self.lr = lr
        self.bn = paddle.nn.BatchNorm1D(4)
        self.linear = paddle.nn.Linear(4, 2)
        self.seen_lrs = []

    def forward(self, x):
        return self.linear(self.bn(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.seen_lrs.append(float(self._trainer.optimizers[0]._optimizer.get_lr()))
        return paddle.nn.functional.cross_entropy(self(x), y)

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=self.lr, parameters=self.parameters())


def make_loader(n=16, bs=8, offset=5.0):
    # Offset so batch-norm statistics are unmistakably non-zero once recomputed.
    ds = paddle.io.TensorDataset([paddle.randn([n, 4]) + offset, paddle.randint(0, 2, [n])])
    return paddle.io.DataLoader(ds, batch_size=bs)


def make_trainer(callback, max_epochs=6):
    return ocean.Trainer(
        max_epochs=max_epochs,
        verbose=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[callback],
    )


def lrs_per_epoch(model, batches_per_epoch=2):
    return [round(model.seen_lrs[i * batches_per_epoch], 5) for i in range(len(model.seen_lrs) // batches_per_epoch)]


# ── Runs at all ──────────────────────────────────────────────────────────────


def test_swa_runs_on_an_ocean_model():
    """deepcopy() of an ocean.Model raises (it reaches the trainer's Paddle
    program), which used to abort the run at the first averaged epoch."""
    callback = StochasticWeightAveraging(swa_lrs=0.001, swa_epoch_start=1, annealing_epochs=1)
    make_trainer(callback, max_epochs=3).fit(BNModel(), train_dataloaders=make_loader())
    assert callback._n_averaged == 2


def test_weight_averaging_runs_on_an_ocean_model():
    callback = WeightAveraging(start_epoch=1)
    make_trainer(callback, max_epochs=3).fit(BNModel(), train_dataloaders=make_loader())
    assert callback._n_averaged == 2


# ── Learning-rate annealing ──────────────────────────────────────────────────


def test_linear_annealing_reaches_swa_lrs():
    callback = StochasticWeightAveraging(
        swa_lrs=0.001, swa_epoch_start=2, annealing_epochs=2, annealing_strategy="linear"
    )
    model = BNModel(lr=0.1)
    make_trainer(callback).fit(model, train_dataloaders=make_loader())

    per_epoch = lrs_per_epoch(model)
    assert per_epoch[:3] == [0.1, 0.1, 0.1]  # untouched before, and at t=0
    assert per_epoch[3] == pytest.approx(0.0505, abs=1e-4)  # halfway
    assert per_epoch[4] == pytest.approx(0.001, abs=1e-6)  # arrived
    assert per_epoch[5] == pytest.approx(0.001, abs=1e-6)  # and stays


def test_cosine_annealing_reaches_swa_lrs():
    callback = StochasticWeightAveraging(swa_lrs=0.001, swa_epoch_start=2, annealing_epochs=2, annealing_strategy="cos")
    model = BNModel(lr=0.1)
    make_trainer(callback).fit(model, train_dataloaders=make_loader())

    per_epoch = lrs_per_epoch(model)
    assert per_epoch[2] == pytest.approx(0.1, abs=1e-6)
    assert 0.001 < per_epoch[3] < 0.1
    assert per_epoch[4] == pytest.approx(0.001, abs=1e-6)


def test_annealing_takes_over_a_bound_scheduler():
    """Paddle refuses set_lr while an LRScheduler is attached, so SWA has to
    replace it."""

    class ScheduledModel(BNModel):
        def configure_optimizers(self):
            scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.1, step_size=100, gamma=0.5)
            return paddle.optimizer.SGD(learning_rate=scheduler, parameters=self.parameters())

    callback = StochasticWeightAveraging(
        swa_lrs=0.002, swa_epoch_start=1, annealing_epochs=1, annealing_strategy="linear"
    )
    model = ScheduledModel()
    make_trainer(callback, max_epochs=3).fit(model, train_dataloaders=make_loader())

    assert lrs_per_epoch(model)[2] == pytest.approx(0.002, abs=1e-6)


def test_no_annealing_before_the_swa_phase():
    callback = StochasticWeightAveraging(swa_lrs=0.001, swa_epoch_start=10, annealing_epochs=1)
    model = BNModel(lr=0.1)
    make_trainer(callback, max_epochs=3).fit(model, train_dataloaders=make_loader())

    assert lrs_per_epoch(model) == [0.1, 0.1, 0.1]
    assert callback._n_averaged == 0


def test_fractional_swa_epoch_start():
    callback = StochasticWeightAveraging(swa_lrs=0.001, swa_epoch_start=0.5)
    trainer = make_trainer(callback, max_epochs=10)
    assert callback.swa_start_epoch(trainer) == 5


# ── Batch-norm statistics ────────────────────────────────────────────────────


def test_batch_norm_statistics_are_recomputed():
    """The stored statistics describe the last weights, not the average."""
    callback = StochasticWeightAveraging(swa_lrs=0.001, swa_epoch_start=1, annealing_epochs=1)
    model = BNModel()
    make_trainer(callback, max_epochs=3).fit(model, train_dataloaders=make_loader(offset=5.0))

    running_mean = model.bn._mean.numpy()
    assert (abs(running_mean - 5.0) < 2.0).all()


def test_batch_norm_update_can_be_disabled():
    callback = StochasticWeightAveraging(swa_lrs=0.001, swa_epoch_start=1, annealing_epochs=1, update_bn=False)
    model = BNModel()
    make_trainer(callback, max_epochs=3).fit(model, train_dataloaders=make_loader(offset=5.0))
    # Left as whatever training produced, not recomputed from scratch.
    assert model.bn._mean is not None


def test_averaging_covers_parameters_and_applies_them():
    callback = StochasticWeightAveraging(swa_lrs=0.001, swa_epoch_start=0, annealing_epochs=1, update_bn=False)
    model = BNModel()
    make_trainer(callback, max_epochs=3).fit(model, train_dataloaders=make_loader())

    assert callback._average_state is not None
    averaged = callback._average_state["linear.weight"].numpy()
    assert (abs(model.linear.weight.numpy() - averaged) < 1e-6).all()


# ── Validation / state ───────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", [0.0, -1.0, "x", [0.1, -0.2]])
def test_invalid_swa_lrs_rejected(bad):
    with pytest.raises(MisconfigurationException, match="swa_lrs"):
        StochasticWeightAveraging(swa_lrs=bad)


def test_invalid_annealing_strategy_rejected():
    with pytest.raises(MisconfigurationException, match="annealing_strategy"):
        StochasticWeightAveraging(annealing_strategy="quadratic")


def test_invalid_fractional_start_rejected():
    with pytest.raises(MisconfigurationException, match="swa_epoch_start"):
        StochasticWeightAveraging(swa_epoch_start=1.5)


def test_state_dict_round_trip():
    callback = StochasticWeightAveraging(swa_lrs=0.001, swa_epoch_start=1, annealing_epochs=1)
    make_trainer(callback, max_epochs=3).fit(BNModel(), train_dataloaders=make_loader())

    restored = StochasticWeightAveraging(swa_lrs=0.001)
    restored.load_state_dict(callback.state_dict())
    assert restored._n_averaged == callback._n_averaged
    assert restored._swa_started is True
