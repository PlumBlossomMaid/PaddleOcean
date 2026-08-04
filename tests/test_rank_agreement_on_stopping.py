"""Tests that the stop decision is agreed across ranks.

A rank deciding on its own does not stop training early — it hangs it: the rank
that leaves the loop never reaches the next collective the others are waiting
at. Both callbacks that can end a run therefore have to reduce their decision.
"""

import paddle
import pytest

import ocean
from ocean.callbacks.early_stopping import EarlyStopping
from ocean.callbacks.timer import Timer


class Recorder:
    """Wraps the real strategy's two reduction entry points.

    Only those are replaced: swapping the whole strategy out takes the rest of
    the trainer with it.
    """

    def __init__(self, strategy, answer=None):
        self.strategy = strategy
        self.answer = answer
        self.boolean_calls = []
        self.broadcast_calls = []
        strategy.reduce_boolean_decision = self._reduce_boolean_decision
        strategy.broadcast = self._broadcast

    def _reduce_boolean_decision(self, decision, all=True):
        self.boolean_calls.append((decision, all))
        return decision if self.answer is None else self.answer

    def _broadcast(self, obj, src=0):
        self.broadcast_calls.append(obj)
        return obj if self.answer is None else self.answer


class Model(ocean.Model):
    def __init__(self, val_losses):
        super().__init__()
        self.linear = paddle.nn.Linear(4, 2)
        self.val_losses = list(val_losses)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return paddle.nn.functional.cross_entropy(self(x), y)

    def validation_step(self, batch, batch_idx):
        epoch = min(self.current_epoch, len(self.val_losses) - 1)
        self.log("val_loss", float(self.val_losses[epoch]))

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.1, parameters=self.parameters())


def make_loader(n=8, bs=8):
    ds = paddle.io.TensorDataset([paddle.randn([n, 4]), paddle.randint(0, 2, [n])])
    return paddle.io.DataLoader(ds, batch_size=bs)


def run(callback, val_losses=(1.0, 1.0, 1.0, 1.0), max_epochs=4, answer=None):
    trainer = ocean.Trainer(
        max_epochs=max_epochs,
        verbose=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[callback],
    )
    recorder = Recorder(trainer.strategy, answer=answer)
    trainer.fit(Model(val_losses), train_dataloaders=make_loader(), val_dataloaders=make_loader())
    return trainer, recorder


# ── EarlyStopping ────────────────────────────────────────────────────────────


def test_early_stopping_reduces_its_decision():
    _, spy = run(EarlyStopping(monitor="val_loss", patience=1))
    assert spy.boolean_calls, "the decision was never reduced across ranks"
    assert all(all_flag is False for _, all_flag in spy.boolean_calls), "one rank asking to stop must be enough"


def test_the_reduced_answer_wins_over_the_local_one():
    """A rank that does not want to stop still stops if another rank does."""
    trainer, spy = run(EarlyStopping(monitor="val_loss", patience=99), answer=True)
    assert trainer.should_stop is True
    assert spy.boolean_calls[0][0] is False  # locally: keep going


def test_a_local_stop_is_overridden_by_the_reduction():
    trainer, _ = run(EarlyStopping(monitor="val_loss", patience=1), max_epochs=3, answer=False)
    assert trainer.should_stop is False
    assert trainer.current_epoch == 3  # ran to the end


def test_early_stopping_still_stops_on_a_plateau():
    trainer, _ = run(EarlyStopping(monitor="val_loss", patience=1), max_epochs=5)
    assert trainer.should_stop is True
    assert trainer.current_epoch < 5


def test_early_stopping_keeps_going_while_improving():
    trainer, _ = run(EarlyStopping(monitor="val_loss", patience=1), val_losses=(1.0, 0.5, 0.2, 0.1), max_epochs=4)
    assert trainer.should_stop is False
    assert trainer.current_epoch == 4


@pytest.mark.parametrize(
    ("kwargs", "losses"),
    [
        ({"check_finite": True}, (float("nan"),)),
        ({"stopping_threshold": 0.5}, (0.1,)),
        ({"divergence_threshold": 5.0}, (9.0,)),
    ],
)
def test_every_criterion_goes_through_the_reduction(kwargs, losses):
    """Each branch used to set should_stop by itself, so each was a separate
    chance for the ranks to disagree."""
    _, spy = run(EarlyStopping(monitor="val_loss", **kwargs), val_losses=losses, max_epochs=1)
    assert spy.boolean_calls and spy.boolean_calls[0][0] is True


# ── Timer ────────────────────────────────────────────────────────────────────


def test_timer_broadcasts_its_decision():
    trainer, spy = run(Timer(duration=3600), max_epochs=1)
    assert spy.broadcast_calls, "the deadline check was never broadcast"


def test_timer_stops_when_rank_zero_says_so():
    """Clocks drift; rank 0's answer is the one that counts."""
    trainer, _ = run(Timer(duration=3600), max_epochs=2, answer=True)
    assert trainer.should_stop is True
