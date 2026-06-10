"""RichModelSummary callback - prints model summary using rich tables.

Fallback to plain text if rich is not installed.
"""

from typing import Any, Optional

from ocean.callbacks.model_summary import ModelSummary


class RichModelSummary(ModelSummary):
    """Print a rich model summary using the rich library.

    Falls back to plain ModelSummary if rich is not installed.
    """

    def __init__(self, max_depth: int = 1) -> None:
        super().__init__(max_depth)

    def _print_model_summary(self, model: Any) -> None:
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.text import Text

            console = Console()
            table = Table(title="Model Summary", show_header=True, header_style="bold magenta")
            table.add_column("Layer", style="cyan")
            table.add_column("Type", style="green")
            table.add_column("Params", justify="right", style="yellow")

            total = 0
            trainable = 0

            for name, param in model.named_parameters():
                num = param.numel().item() if hasattr(param.numel(), "item") else int(param.numel())
                total += num
                if not param.stop_gradient:
                    trainable += num

            for name, module in model.named_children():
                num = sum(
                    p.numel().item() if hasattr(p.numel(), "item") else int(p.numel())
                    for p in module.parameters()
                )
                table.add_row(name, module.__class__.__name__, f"{num:,}")

            table.add_section()
            table.add_row("Total params", "", f"{total:,}")
            table.add_row("Trainable params", "", f"{trainable:,}")
            table.add_row("Non-trainable params", "", f"{total - trainable:,}")

            console.print(table)
        except ImportError:
            super()._print_model_summary(model)
