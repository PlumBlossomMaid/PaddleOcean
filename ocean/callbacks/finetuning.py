"""Finetuning callbacks — freeze/unfreeze modules and schedule backbone LR.

Adapts the reference BaseFinetuning/BackboneFinetuning design to Paddle. The one
semantic difference that matters: a Paddle optimizer param group's
``learning_rate`` is a *scale multiplier* applied on top of the optimizer's base
learning rate (stored per parameter in ``param.optimize_attr['learning_rate']``),
whereas the reference framework stores an absolute per-group LR. To keep the
user-facing behaviour aligned — where ``backbone_initial_lr`` and the aligned
head LR are absolute learning rates — every absolute LR is converted to a scale
of the optimizer base LR before it is written to the param group.
"""

from typing import Any, Callable, Iterable, Optional, Union

from paddle.nn import Layer
from paddle.nn.layer.norm import _BatchNormBase

from ocean.callbacks.callback import Callback


def multiplicative(epoch: int) -> float:
    return 2.0


def _flatten_modules(modules: Union[Layer, Iterable]) -> list:
    """Flatten a module (or iterable of modules) into its leaf/parameter-bearing layers.

    Mirrors the reference ``flatten_modules``: keep leaf layers plus any parent
    layer that directly owns parameters, so ``parameters(include_sublayers=False)``
    on each returned layer never double-counts a parameter.
    """
    if isinstance(modules, Layer):
        flat = modules.sublayers(include_self=True)
    else:
        flat = []
        for m in modules:
            flat.extend(_flatten_modules(m))
    # Keep leaves, and non-leaf layers that hold parameters of their own.
    return [m for m in flat if not list(m.children()) or m._parameters]


class BaseFinetuning(Callback):
    """Base callback for freeze/unfreeze finetuning schedules.

    Subclasses implement :meth:`freeze_before_training` and
    :meth:`finetune_function`. Freezing is applied at ``on_fit_start`` (Ocean does
    not dispatch a callback ``setup`` hook), and the per-epoch unfreeze schedule
    runs at ``on_train_epoch_start``.
    """

    def __init__(self) -> None:
        self._internal_optimizer_metadata: dict[int, list] = {}

    # ------------------------------------------------------------------
    # Freeze / unfreeze primitives
    # ------------------------------------------------------------------
    @staticmethod
    def flatten_modules(modules: Union[Layer, Iterable]) -> list:
        return _flatten_modules(modules)

    @staticmethod
    def filter_params(modules: Union[Layer, Iterable], train_bn: bool = True, requires_grad: bool = True) -> Iterable:
        """Yield params whose trainable state matches ``requires_grad``.

        ``requires_grad`` is the reference framework's term; in Paddle a trainable
        parameter is one with ``stop_gradient is False``.
        """
        for mod in _flatten_modules(modules):
            if isinstance(mod, _BatchNormBase) and not train_bn:
                continue
            for param in mod.parameters(include_sublayers=False):
                if (not param.stop_gradient) == requires_grad:
                    yield param

    @staticmethod
    def make_trainable(modules: Union[Layer, Iterable]) -> None:
        """Unfreeze the parameters of the provided modules."""
        for module in _flatten_modules(modules):
            if isinstance(module, _BatchNormBase):
                module._use_global_stats = False
            for param in module.parameters(include_sublayers=False):
                param.stop_gradient = False

    @staticmethod
    def freeze_module(module: Layer) -> None:
        """Freeze the parameters of a single module."""
        if isinstance(module, _BatchNormBase):
            module._use_global_stats = True
        for param in module.parameters(include_sublayers=False):
            param.stop_gradient = True

    @staticmethod
    def freeze(modules: Union[Layer, Iterable], train_bn: bool = True) -> None:
        """Freeze the provided modules; optionally leave BatchNorm trainable."""
        for mod in _flatten_modules(modules):
            if isinstance(mod, _BatchNormBase) and train_bn:
                BaseFinetuning.make_trainable(mod)
            else:
                BaseFinetuning.freeze_module(mod)

    # ------------------------------------------------------------------
    # Optimizer param-group management
    # ------------------------------------------------------------------
    @staticmethod
    def _raw_optimizer(optimizer: Any) -> Any:
        """Unwrap an OceanOptimizer to the underlying paddle optimizer."""
        return getattr(optimizer, "_optimizer", optimizer)

    @staticmethod
    def filter_on_optimizer(optimizer: Any, params: Iterable) -> list:
        """Return only params not already present in the optimizer's param groups."""
        raw = BaseFinetuning._raw_optimizer(optimizer)
        existing = set()
        for group in raw._param_groups:
            for p in group["params"]:
                existing.add(id(p))
        return [p for p in params if id(p) not in existing]

    @staticmethod
    def _base_lr(optimizer: Any) -> float:
        raw = BaseFinetuning._raw_optimizer(optimizer)
        try:
            return float(raw.get_lr())
        except Exception:
            return 1.0

    @staticmethod
    def unfreeze_and_add_param_group(
        modules: Union[Layer, Iterable],
        optimizer: Any,
        lr: Optional[float] = None,
        initial_denom_lr: float = 10.0,
        train_bn: bool = True,
    ) -> None:
        """Unfreeze ``modules`` and add their params to ``optimizer`` as a new group.

        ``lr`` (when given) is an *absolute* learning rate; it is converted to a
        Paddle scale of the optimizer base LR. When ``lr`` is None the new group
        starts at ``base_lr / initial_denom_lr``.
        """
        BaseFinetuning.make_trainable(modules)
        base_lr = BaseFinetuning._base_lr(optimizer)
        denom = initial_denom_lr if lr is None else 1.0
        target_lr = (base_lr if lr is None else float(lr)) / denom
        scale = target_lr / base_lr if base_lr else 1.0

        params = list(BaseFinetuning.filter_params(modules, train_bn=train_bn, requires_grad=True))
        params = BaseFinetuning.filter_on_optimizer(optimizer, params)
        if params:
            raw = BaseFinetuning._raw_optimizer(optimizer)
            # ``grad_clip`` must be set explicitly: a group added without it is left
            # with a malformed clip entry that breaks ``optimizer.step()``.
            raw._add_param_group({
                "params": params,
                "learning_rate": scale,
                "weight_decay": 0.0,
                "grad_clip": raw._grad_clip,
            })

    @staticmethod
    def _group_scale_for_abs_lr(optimizer: Any, abs_lr: float) -> float:
        base_lr = BaseFinetuning._base_lr(optimizer)
        return abs_lr / base_lr if base_lr else 1.0

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------
    def on_fit_start(self, trainer: Any, model: Any) -> None:
        # Ocean has no callback `setup` dispatch, so freezing happens here.
        self.freeze_before_training(model)

    def on_train_epoch_start(self, trainer: Any, model: Any) -> None:
        for opt_idx, optimizer in enumerate(getattr(trainer, "optimizers", None) or []):
            self.finetune_function(model, trainer.current_epoch, optimizer)

    def freeze_before_training(self, model: Any) -> None:
        raise NotImplementedError("Override `freeze_before_training` with your freeze logic.")

    def finetune_function(self, model: Any, epoch: int, optimizer: Any) -> None:
        raise NotImplementedError("Override `finetune_function` with your unfreeze logic.")


class BackboneFinetuning(BaseFinetuning):
    r"""Finetune a backbone on a learning-rate schedule.

    The model must expose an ``nn.Layer`` ``backbone`` attribute. The backbone is
    frozen at the start of training and unfrozen at
    ``unfreeze_backbone_at_epoch`` with an initial LR, which is then scaled each
    epoch by ``lambda_func``. When ``should_align`` is set and the scheduled
    backbone LR would exceed the current (head) LR, it is clamped to the head LR.

    Args:
        unfreeze_backbone_at_epoch: Epoch at which the backbone is unfrozen.
        lambda_func: Per-epoch multiplicative schedule for the backbone LR.
        backbone_initial_ratio_lr: Backbone starts at this fraction of the head LR.
        backbone_initial_lr: Absolute initial backbone LR (overrides the ratio).
        should_align: Clamp the backbone LR to the head LR once it catches up.
        initial_denom_lr: When unfreezing, initial LR is ``head_lr / initial_denom_lr``.
        train_bn: Whether BatchNorm layers are trainable.
        verbose: Print the head/backbone LR each scheduled epoch.
        rounding: Digits for the printed LR.
    """

    def __init__(
        self,
        unfreeze_backbone_at_epoch: int = 10,
        lambda_func: Callable = multiplicative,
        backbone_initial_ratio_lr: float = 0.1,
        backbone_initial_lr: Optional[float] = None,
        should_align: bool = True,
        initial_denom_lr: float = 10.0,
        train_bn: bool = True,
        verbose: bool = False,
        rounding: int = 12,
        # Backwards-compatible alias for the previous ``backbone_initial_ratio`` name.
        backbone_initial_ratio: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.unfreeze_backbone_at_epoch = unfreeze_backbone_at_epoch
        self.lambda_func = lambda_func
        if backbone_initial_ratio is not None:
            backbone_initial_ratio_lr = backbone_initial_ratio
        self.backbone_initial_ratio_lr = backbone_initial_ratio_lr
        self.backbone_initial_lr = backbone_initial_lr
        self.should_align = should_align
        self.initial_denom_lr = initial_denom_lr
        self.train_bn = train_bn
        self.verbose = verbose
        self.rounding = rounding
        self.previous_backbone_lr: Optional[float] = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "internal_optimizer_metadata": self._internal_optimizer_metadata,
            "previous_backbone_lr": self.previous_backbone_lr,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.previous_backbone_lr = state_dict.get("previous_backbone_lr")
        self._internal_optimizer_metadata = state_dict.get("internal_optimizer_metadata", {})

    def on_fit_start(self, trainer: Any, model: Any) -> None:
        backbone = getattr(model, "backbone", None)
        if not isinstance(backbone, Layer):
            raise ValueError("BackboneFinetuning requires the model to have an `nn.Layer` `backbone` attribute.")
        self.freeze_before_training(model)

    def freeze_before_training(self, model: Any) -> None:
        self.freeze(model.backbone, train_bn=self.train_bn)

    def _set_backbone_group_lr(self, optimizer: Any, abs_lr: float) -> None:
        """Write an absolute backbone LR to the last (backbone) param group as a scale."""
        raw = self._raw_optimizer(optimizer)
        scale = self._group_scale_for_abs_lr(optimizer, abs_lr)
        group = raw._param_groups[-1]
        group["learning_rate"] = scale
        for p in group["params"]:
            p.optimize_attr["learning_rate"] = scale

    def finetune_function(self, model: Any, epoch: int, optimizer: Any) -> None:
        if epoch == self.unfreeze_backbone_at_epoch:
            current_lr = self._base_lr(optimizer)
            initial_backbone_lr = (
                self.backbone_initial_lr
                if self.backbone_initial_lr is not None
                else current_lr * self.backbone_initial_ratio_lr
            )
            self.previous_backbone_lr = initial_backbone_lr
            self.unfreeze_and_add_param_group(
                model.backbone,
                optimizer,
                initial_backbone_lr,
                train_bn=self.train_bn,
                initial_denom_lr=self.initial_denom_lr,
            )
            if self.verbose:
                print(
                    f"Current lr: {round(current_lr, self.rounding)}, Backbone lr: {round(initial_backbone_lr, self.rounding)}"
                )

        elif epoch > self.unfreeze_backbone_at_epoch:
            current_lr = self._base_lr(optimizer)
            next_backbone_lr = self.lambda_func(epoch + 1) * (self.previous_backbone_lr or 0.0)
            next_backbone_lr = current_lr if (self.should_align and next_backbone_lr > current_lr) else next_backbone_lr
            self._set_backbone_group_lr(optimizer, next_backbone_lr)
            self.previous_backbone_lr = next_backbone_lr
            if self.verbose:
                print(
                    f"Current lr: {round(current_lr, self.rounding)}, Backbone lr: {round(next_backbone_lr, self.rounding)}"
                )
