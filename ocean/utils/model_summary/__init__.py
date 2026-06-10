"""Model summary data structures.

Provides the Summary data type used by ModelSummary callback.
"""

from typing import Any, Optional


class ModelSummary:
    """Summary of a model's layers, parameters, and FLOPs.

    Args:
        summary_data: List of (layer_name, details_list) tuples.
        total_parameters: Total number of parameters.
        trainable_parameters: Number of trainable parameters.
        model_size: Model size in bytes.
    """

    def __init__(
        self,
        summary_data: list[tuple[str, list[str]]],
        total_parameters: int = 0,
        trainable_parameters: int = 0,
        model_size: float = 0.0,
    ) -> None:
        self.summary_data = summary_data
        self.total_parameters = total_parameters
        self.trainable_parameters = trainable_parameters
        self.model_size = model_size
        self.total_non_trainable_parameters = total_parameters - trainable_parameters

    def __repr__(self) -> str:
        return (
            f"ModelSummary(total_params={self.total_parameters}, "
            f"trainable_params={self.trainable_parameters})"
        )


def summarize(model: Any, max_depth: int = 1) -> ModelSummary:
    """Analyze a model and produce a ModelSummary.

    Args:
        model: The model to summarize.
        max_depth: Maximum depth to recurse into submodules.

    Returns:
        A ModelSummary instance.
    """
    total_params = 0
    trainable_params = 0
    layers = []

    for name, param in model.named_parameters():
        num = param.numel().item() if hasattr(param.numel(), "item") else int(param.numel())
        total_params += num
        if not param.stop_gradient:
            trainable_params += num

    # Estimate model size (float32 = 4 bytes per parameter)
    model_size = total_params * 4.0

    # Collect layer info
    for name, module in model.named_children():
        num = sum(
            p.numel().item() if hasattr(p.numel(), "item") else int(p.numel())
            for p in module.parameters()
        )
        layers.append((name, [module.__class__.__name__, f"{num:,}"]))

    return ModelSummary(
        summary_data=layers,
        total_parameters=total_params,
        trainable_parameters=trainable_params,
        model_size=model_size,
    )
