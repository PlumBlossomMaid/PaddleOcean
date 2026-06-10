"""BackboneFinetuning callback - gradually unfreezes backbone layers for finetuning."""

from typing import Any, Optional

from ocean.callbacks.callback import Callback


class BackboneFinetuning(Callback):
    """Gradually unfreeze backbone layers during finetuning.

    Args:
        unfreeze_backbone_at_epoch: Epoch at which to start unfreezing.
        backbone_initial_ratio: Initial learning rate ratio for backbone.
        backbone_initial_lr: Optional fixed initial LR for backbone.
        should_align: If True, align backbone LR with head LR.
        train_bn: If True, train batch norm layers.
    """

    def __init__(
        self,
        unfreeze_backbone_at_epoch: int = 5,
        backbone_initial_ratio: float = 0.1,
        backbone_initial_lr: Optional[float] = None,
        should_align: bool = True,
        train_bn: bool = True,
    ) -> None:
        self.unfreeze_backbone_at_epoch = unfreeze_backbone_at_epoch
        self.backbone_initial_ratio = backbone_initial_ratio
        self.backbone_initial_lr = backbone_initial_lr
        self.should_align = should_align
        self.train_bn = train_bn

    def on_train_epoch_start(self, trainer: Any, model: Any) -> None:
        if trainer.current_epoch >= self.unfreeze_backbone_at_epoch:
            self._unfreeze_backbone(model)

    def _unfreeze_backbone(self, model: Any) -> None:
        for param in model.parameters():
            param.stop_gradient = False
