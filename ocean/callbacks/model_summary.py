from ocean.callbacks.callback import Callback


class ModelSummary(Callback):
    def __init__(self, max_depth: int = 1):
        self.max_depth = max_depth

    def on_fit_start(self, trainer, pl_module):
        self._summarize(pl_module)

    def _summarize(self, model):
        print(self._get_summary(model))

    def _get_summary(self, model):
        lines = ["", "  | Name  | Type  | Params  | In size  | Out size  |"]
        total_params = 0
        for name, layer in model.named_sublayers():
            if name == "":
                continue
            params = sum(p.numel() for p in layer.parameters() if not p.stop_gradient)
            total_params += params
            lines.append(f"  | {name} | {layer.__class__.__name__} | {params:,} | - | - |")

        # Add total
        lines.append(f"  {'─' * 50}")
        lines.append(f"  Total params: {total_params:,}")
        lines.append(f"  Trainable params: {total_params:,}")
        lines.append("")
        return "\n".join(lines)
