from ocean.callbacks.callback import Callback


class ModelSummary(Callback):
    def __init__(self, max_depth: int = 1):
        self.max_depth = max_depth

    def on_fit_start(self, trainer, model):
        self._summarize(model)

    def _summarize(self, model):
        print(self._get_summary(model))

    def _get_summary(self, model):
        lines = ["", "  | Name  | Type  | Params  | In size  | Out size  |"]
        total_params = 0
        trainable_params = 0
        for name, layer in self._named_layers_depth(model, self.max_depth):
            params = sum(p.numel() for p in layer.parameters() if not p.stop_gradient)
            non_trainable = sum(p.numel() for p in layer.parameters() if p.stop_gradient)
            total_params += params + non_trainable
            trainable_params += params
            lines.append(f"  | {name} | {layer.__class__.__name__} | {params + non_trainable:,} | - | - |")

        non_trainable = total_params - trainable_params
        lines.append(f"  {'─' * 50}")
        lines.append(f"  Total params: {total_params:,}")
        lines.append(f"  Trainable params: {trainable_params:,}")
        lines.append(f"  Non-trainable params: {non_trainable:,}")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _named_layers_depth(model, max_depth: int, prefix: str = "", current_depth: int = 0):
        """Yield (name, layer) respecting max_depth (ocean-compatible)."""
        if current_depth > max_depth:
            return
        for name, child in model.named_children():
            full = f"{prefix}.{name}" if prefix else name
            yield full, child
            yield from ModelSummary._named_layers_depth(child, max_depth, full, current_depth + 1)
