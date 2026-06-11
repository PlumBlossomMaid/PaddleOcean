"""ModelSummary callback - prints a summary of the model before training."""

from typing import Any

from ocean.callbacks.callback import Callback


class ModelSummary(Callback):
    """Print a summary of the model architecture before training.

    Args:
        max_depth: Maximum depth of nested layers to show. 0 disables summary.
    """

    def __init__(self, max_depth: int = 1) -> None:
        self.max_depth = max_depth

    def on_fit_start(self, trainer: Any, model: Any) -> None:
        if self.max_depth == 0:
            return
        if not getattr(trainer, "is_global_zero", True):
            return

        print(f"\n  {'='*60}")
        print(f"  {'Model Summary':^60}")
        print(f"  {'='*60}")
        self._print_model_summary(model)
        print(f"  {'='*60}\n")

    def _print_model_summary(self, model: Any) -> None:
        total_params = 0
        trainable_params = 0
        layers = []

        for name, param in model.named_parameters():
            num = param.numel().item() if hasattr(param.numel(), "item") else int(param.numel())
            total_params += num
            if not param.stop_gradient:
                trainable_params += num

        for name, module in model.named_children():
            num = sum(p.numel().item() if hasattr(p.numel(), "item") else int(p.numel()) for p in module.parameters())
            layers.append((name, module.__class__.__name__, num))

        for name, cls_name, num in layers:
            print(f"  {name:30s} {cls_name:25s} {num:>8,} params")

        print(f"  {'─'*60}")
        print(f"  {'Total params':30s} {total_params:>28,}")
        print(f"  {'Trainable params':30s} {trainable_params:>28,}")
        print(f"  {'Non-trainable params':30s} {total_params - trainable_params:>28,}")
