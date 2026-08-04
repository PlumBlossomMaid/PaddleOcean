"""StochasticWeightAveraging callback - averages model weights for better generalization."""

import math
from typing import Any, Optional, Union

import paddle

from ocean.callbacks.callback import Callback
from ocean.utils import MisconfigurationException
from ocean.utils.rank_zero import rank_zero_warn

_BATCH_NORM_TYPES = (
    paddle.nn.BatchNorm1D,
    paddle.nn.BatchNorm2D,
    paddle.nn.BatchNorm3D,
    paddle.nn.SyncBatchNorm,
)


class StochasticWeightAveraging(Callback):
    """Averaging model weights with SWA for improved generalization.

    From ``swa_epoch_start`` onwards the weights of every epoch are averaged and
    the learning rate is annealed to ``swa_lrs``. SWA is both halves: the
    averaging is what improves generalization, and the flat, deliberately
    not-tiny learning rate is what makes the individual weights worth averaging.

    Args:
        swa_lrs: Learning rate to anneal to for the SWA phase. A single value is
            applied to every optimizer; pass a list to give each its own.
        swa_epoch_start: Epoch to start SWA (float=fraction of max_epochs,
            int=epoch number).
        annealing_epochs: Number of epochs over which the learning rate is
            annealed from its value at SWA start to ``swa_lrs``.
        annealing_strategy: ``'cos'`` or ``'linear'``.
        update_bn: Whether to recompute batch-norm statistics over the training
            data once the averaged weights are applied. The stored running
            statistics describe the *last* weights, not the averaged ones, so
            without this a model with batch norm evaluates against stale stats.
    """

    def __init__(
        self,
        swa_lrs: Union[float, list[float]] = 1e-3,
        swa_epoch_start: Union[int, float] = 0.8,
        annealing_epochs: int = 10,
        annealing_strategy: str = "cos",
        update_bn: bool = True,
    ) -> None:
        wrong_float = isinstance(swa_lrs, float) and swa_lrs <= 0
        wrong_list = isinstance(swa_lrs, list) and not all(isinstance(lr, float) and lr > 0 for lr in swa_lrs)
        if not isinstance(swa_lrs, (float, list)) or wrong_float or wrong_list:
            raise MisconfigurationException("The `swa_lrs` should be a positive float, or a list of positive floats")
        if annealing_strategy not in ("cos", "linear"):
            raise MisconfigurationException(
                f"The `annealing_strategy` should be 'cos' or 'linear', got {annealing_strategy!r}"
            )
        if isinstance(swa_epoch_start, float) and not 0.0 <= swa_epoch_start <= 1.0:
            raise MisconfigurationException(
                f"A float `swa_epoch_start` is a fraction of max_epochs and must be in [0, 1], got {swa_epoch_start}"
            )

        self.swa_lrs = swa_lrs
        self.swa_epoch_start = swa_epoch_start
        self.annealing_epochs = annealing_epochs
        self.annealing_strategy = annealing_strategy
        self.update_bn = update_bn
        self._average_state: Optional[dict[str, Any]] = None
        self._n_averaged = 0
        self._initial_lrs: list[Optional[float]] = []
        self._swa_started = False

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    def swa_start_epoch(self, trainer: Any) -> int:
        """Resolve ``swa_epoch_start`` against the run's length."""
        if isinstance(self.swa_epoch_start, float):
            return int(self.swa_epoch_start * (trainer.max_epochs or 1))
        return self.swa_epoch_start

    def on_fit_start(self, trainer: Any, model: Any) -> None:
        self._n_averaged = 0
        self._average_state = None
        self._initial_lrs = []
        self._swa_started = False

    def on_train_epoch_start(self, trainer: Any, model: Any) -> None:
        """Anneal the learning rate once the SWA phase has begun."""
        epoch_in_swa = trainer.current_epoch - self.swa_start_epoch(trainer)
        if epoch_in_swa < 0:
            return

        optimizers = list(getattr(trainer, "optimizers", None) or [])
        if not optimizers:
            return

        if not self._swa_started:
            self._swa_started = True
            self._initial_lrs = [self._current_lr(opt) for opt in optimizers]

        for index, (optimizer, initial_lr) in enumerate(zip(optimizers, self._initial_lrs)):
            if initial_lr is None:
                continue
            self._set_lr(optimizer, self._annealed_lr(initial_lr, epoch_in_swa, index))

    def _annealed_lr(self, initial_lr: float, epoch_in_swa: int, index: int) -> float:
        """Interpolate from the learning rate at SWA start toward ``swa_lrs``."""
        target = self.swa_lrs[index] if isinstance(self.swa_lrs, list) else self.swa_lrs
        if self.annealing_epochs <= 0:
            return target
        t = min(1.0, epoch_in_swa / self.annealing_epochs)
        if self.annealing_strategy == "cos":
            # 1 at t=0 falling to 0 at t=1, following a cosine.
            return target + (initial_lr - target) * (1 + math.cos(math.pi * t)) / 2
        return initial_lr + (target - initial_lr) * t

    @staticmethod
    def _current_lr(optimizer: Any) -> Optional[float]:
        raw = getattr(optimizer, "_optimizer", optimizer)
        try:
            return float(raw.get_lr())
        except Exception:
            return None

    @staticmethod
    def _set_lr(optimizer: Any, value: float) -> None:
        """Take over the learning rate of a Paddle optimizer.

        Paddle keeps the schedule *inside* the optimizer and refuses ``set_lr``
        while an ``LRScheduler`` is attached, so SWA has to replace it with a
        plain value first. The reference framework swaps in an SWA scheduler
        object at this point — the same handover, expressed the way each
        framework stores learning rates.
        """
        raw = getattr(optimizer, "_optimizer", optimizer)
        if isinstance(getattr(raw, "_learning_rate", None), paddle.optimizer.lr.LRScheduler):
            raw._learning_rate = value
            return
        try:
            raw.set_lr(value)
        except Exception:  # pragma: no cover - optimizer without set_lr
            rank_zero_warn(f"Could not set the SWA learning rate on {type(raw).__name__}.")

    # ------------------------------------------------------------------
    # Averaging
    # ------------------------------------------------------------------

    def on_train_epoch_end(self, trainer: Any, model: Any) -> None:
        if trainer.current_epoch < self.swa_start_epoch(trainer):
            return
        if self._average_state is None:
            self._average_state = {k: v.clone() for k, v in model.state_dict().items()}
            self._n_averaged = 1
        else:
            self._update_average(model)
            self._n_averaged += 1

    def on_train_end(self, trainer: Any, model: Any) -> None:
        if self._average_state is None or self._n_averaged == 0:
            return
        model.set_state_dict(self._average_state)
        if self.update_bn:
            self._update_bn(trainer, model)

    def _update_average(self, model: Any) -> None:
        """Running mean of the parameters; buffers track the latest weights.

        Averaging is over the *parameters* only — batch-norm running statistics
        are buffers describing activations, and the average of two epochs' worth
        of statistics describes neither model. ``_update_bn`` recomputes them at
        the end instead.
        """
        if self._average_state is None:
            return
        parameter_names = {name for name, _ in model.named_parameters()}
        n = self._n_averaged + 1
        with paddle.no_grad():
            for name, value in model.state_dict().items():
                if name not in self._average_state:
                    continue
                if name in parameter_names and paddle.is_floating_point(value):
                    averaged = self._average_state[name]
                    averaged.set_value(averaged * (n - 1) / n + value * (1 / n))
                else:
                    self._average_state[name] = value.clone()

    # ------------------------------------------------------------------
    # Batch-norm statistics
    # ------------------------------------------------------------------

    def _update_bn(self, trainer: Any, model: Any) -> None:
        """Recompute batch-norm running statistics for the averaged weights.

        The running mean/variance stored in the model describe the activations
        of the final weights, not of the average, so evaluating the averaged
        model against them is simply wrong. One forward pass over the training
        data rebuilds them.
        """
        bn_layers = [m for m in model.sublayers() if isinstance(m, _BATCH_NORM_TYPES)]
        if not bn_layers:
            return
        loader = getattr(trainer, "train_dataloader", None)
        if loader is None:
            rank_zero_warn("Skipping the SWA batch-norm update: no training dataloader is attached.")
            return

        momenta = {layer: layer._momentum for layer in bn_layers}
        for layer in bn_layers:
            layer._mean.set_value(paddle.zeros_like(layer._mean))
            layer._variance.set_value(paddle.ones_like(layer._variance))

        was_training = model.training
        model.train()
        device = trainer._resolve_device()
        seen = 0
        with paddle.no_grad():
            for batch in loader:
                seen += 1
                # Paddle updates as `running = m * running + (1 - m) * batch`,
                # so this momentum makes it a cumulative average over batches.
                for layer in bn_layers:
                    layer._momentum = 1.0 - 1.0 / seen
                batch = trainer._move_to_device(batch, device)
                inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                model(inputs)

        for layer, momentum in momenta.items():
            layer._momentum = momentum
        if not was_training:
            model.eval()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "n_averaged": self._n_averaged,
            "swa_started": self._swa_started,
            "initial_lrs": self._initial_lrs,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._n_averaged = state_dict.get("n_averaged", 0)
        self._swa_started = state_dict.get("swa_started", False)
        self._initial_lrs = state_dict.get("initial_lrs", [])
