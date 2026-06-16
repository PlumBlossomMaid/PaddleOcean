"""RichModelSummary callback - prints model summary using rich tables.

Fallback to plain text if rich is not installed.
"""

from typing import Any

from ocean.callbacks.model_summary import ModelSummary


class RichModelSummary(ModelSummary):
    """Print a rich model summary using the rich library.

    Falls back to plain ModelSummary if rich is not installed.
    """

    def __init__(self, max_depth: int = 1) -> None:
        super().__init__(max_depth)

    def _summarize(self, model: Any) -> None:
        """Override to use rich table instead of plain text (fixes method name mismatch)."""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title="Model Summary", show_header=True, header_style="bold magenta")
            table.add_column("Layer", style="cyan")
            table.add_column("Type", style="green")
            table.add_column("Params", justify="right", style="yellow")

            total = 0
            trainable = 0

            for name, layer in self._named_layers_depth(model, self.max_depth):
                params = sum(
                    p.numel().item() if hasattr(p.numel(), "item") else int(p.numel())
                    for p in layer.parameters()
                )
                total += params
                if any(not p.stop_gradient for p in layer.parameters()):
                    trainable += sum(
                        p.numel().item() if hasattr(p.numel(), "item") else int(p.numel())
                        for p in layer.parameters() if not p.stop_gradient
                    )

                table.add_row(name, layer.__class__.__name__, f"{params:,}")

            table.add_section()
            table.add_row("Total params", "", f"{total:,}")
            table.add_row("Trainable params", "", f"{trainable:,}")
            table.add_row("Non-trainable params", "", f"{total - trainable:,}")

            console.print(table)
        except ImportError:
            # Fallback to plain text
            super()._summarize(model)
