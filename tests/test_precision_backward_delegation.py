"""Tests for manual optimization going through the strategy / precision plugin.

Covers:
- manual_backward routes through the precision plugin (loss scaling applies)
- manual_backward rejects being called in automatic optimization
- OceanOptimizer.step() routes through the precision plugin
- an overridden Model.backward is honoured by the precision plugin
- Strategy.backward threads pre_backward's return value into backward
"""

import paddle
import pytest

import ocean
from ocean.core.optimizer import OceanOptimizer
from ocean.model import Model
from ocean.plugins.precision.precision import Precision

# ── Spy precision plugin ─────────────────────────────────────────────────────


class SpyPrecision(Precision):
    """Records the calls the training loop makes, and scales the loss by `scale`."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__("32-true")
        self.scale = scale
        self.pre_backward_calls = 0
        self.backward_calls = 0
        self.optimizer_step_calls = 0

    def pre_backward(self, tensor, module):
        self.pre_backward_calls += 1
        return tensor * self.scale

    def backward(self, tensor, model, *args, **kwargs):
        self.backward_calls += 1
        return super().backward(tensor, model, *args, **kwargs)

    def optimizer_step(self, optimizer, **kwargs):
        self.optimizer_step_calls += 1
        return super().optimizer_step(optimizer, **kwargs)


class ManualModel(Model):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False
        self.linear = paddle.nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        x, y = batch
        loss = paddle.nn.functional.cross_entropy(self(x), y)
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        return loss

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())


def make_loader(n=16, bs=8):
    ds = paddle.io.TensorDataset([paddle.randn([n, 10]), paddle.randint(0, 2, [n])])
    return paddle.io.DataLoader(ds, batch_size=bs)


def make_trainer(**kwargs):
    kwargs.setdefault("max_epochs", 1)
    kwargs.setdefault("verbose", 0)
    kwargs.setdefault("logger", False)
    kwargs.setdefault("enable_checkpointing", False)
    kwargs.setdefault("enable_progress_bar", False)
    return ocean.Trainer(**kwargs)


def run_manual(model, precision_plugin, **trainer_kwargs):
    trainer = make_trainer(**trainer_kwargs)
    trainer.strategy._precision_plugin = precision_plugin
    trainer.fit(model, train_dataloaders=make_loader())
    return trainer


# ── manual_backward ──────────────────────────────────────────────────────────


def test_manual_backward_goes_through_the_precision_plugin():
    spy = SpyPrecision()
    model = ManualModel()
    run_manual(model, spy)

    assert spy.pre_backward_calls == 2
    assert spy.backward_calls == 2


def test_manual_backward_applies_loss_scaling():
    """The scaled loss must be the one differentiated, not the raw one.

    Same weights and same batch both times, no optimizer step in between, so the
    only difference is whether the plugin's scaling reached the graph.
    """
    model = ManualModel()
    trainer = make_trainer()
    trainer.strategy._precision_plugin = SpyPrecision(scale=4.0)
    model._trainer = trainer

    x = paddle.ones([4, 10])
    y = paddle.zeros([4], dtype="int64")

    model.manual_backward(paddle.nn.functional.cross_entropy(model(x), y))
    scaled_grad = float(model.linear.weight.grad.abs().sum())

    model.clear_gradients()
    paddle.nn.functional.cross_entropy(model(x), y).backward()
    raw_grad = float(model.linear.weight.grad.abs().sum())

    assert scaled_grad == pytest.approx(raw_grad * 4.0, rel=1e-4)


def test_manual_backward_rejected_in_automatic_optimization():
    class Wrong(ManualModel):
        def __init__(self):
            super().__init__()
            self.automatic_optimization = True

        def training_step(self, batch, batch_idx):
            x, y = batch
            loss = paddle.nn.functional.cross_entropy(self(x), y)
            self.manual_backward(loss)
            return loss

    with pytest.raises(RuntimeError, match="manual optimization"):
        make_trainer().fit(Wrong(), train_dataloaders=make_loader())


def test_manual_backward_without_trainer_still_works():
    model = ManualModel()
    loss = paddle.nn.functional.cross_entropy(model(paddle.randn([4, 10])), paddle.randint(0, 2, [4]))
    model.manual_backward(loss)
    assert model.linear.weight.grad is not None


# ── OceanOptimizer.step ──────────────────────────────────────────────────────


def test_optimizer_step_goes_through_the_precision_plugin():
    spy = SpyPrecision()
    run_manual(ManualModel(), spy)
    assert spy.optimizer_step_calls == 2


def test_optimizer_step_without_strategy_falls_back_to_paddle():
    linear = paddle.nn.Linear(4, 2)
    opt = OceanOptimizer(paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters()))
    linear(paddle.randn([2, 4])).sum().backward()
    before = linear.weight.numpy().copy()
    opt.step()
    assert not (linear.weight.numpy() == before).all()


def test_optimizer_zero_grad():
    linear = paddle.nn.Linear(4, 2)
    opt = OceanOptimizer(paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters()))
    linear(paddle.randn([2, 4])).sum().backward()
    assert linear.weight.grad is not None
    opt.zero_grad()
    assert linear.weight.grad is None or float(linear.weight.grad.abs().sum()) == 0.0


# ── Model.backward override ──────────────────────────────────────────────────


def test_model_backward_override_is_honoured():
    """The precision plugin must delegate to the model, not call tensor.backward()."""

    class OverridingModel(ManualModel):
        def __init__(self):
            super().__init__()
            self.backward_calls = 0

        def backward(self, loss, *args, **kwargs):
            self.backward_calls += 1
            loss.backward(*args, **kwargs)

    model = OverridingModel()
    run_manual(model, SpyPrecision())
    assert model.backward_calls == 2


def test_automatic_optimization_also_uses_the_plugin():
    class AutoModel(Model):
        def __init__(self):
            super().__init__()
            self.linear = paddle.nn.Linear(10, 2)

        def forward(self, x):
            return self.linear(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            return paddle.nn.functional.cross_entropy(self(x), y)

        def configure_optimizers(self):
            return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())

    spy = SpyPrecision()
    trainer = make_trainer()
    trainer.strategy._precision_plugin = spy
    trainer.fit(AutoModel(), train_dataloaders=make_loader())

    assert spy.pre_backward_calls == 2
    assert spy.backward_calls == 2
    assert spy.optimizer_step_calls == 2


# ── DummyLogger experiment indexing ──────────────────────────────────────────


def test_dummy_experiment_supports_indexing():
    """`logger.experiment[0].add_image(...)` must not blow up while logging is off."""
    from ocean.loggers import DummyLogger

    logger = DummyLogger()
    logger.experiment.add_scalar("x", 1.0)
    logger.experiment[0].add_image("img", None)
    logger.experiment[0] = "ignored"
    assert logger[0] is logger
