"""ModelSummary callback - prints a layer table with parameter and shape info."""

from typing import Any

import paddle

from ocean.callbacks.callback import Callback

UNKNOWN_SIZE = "-"


def _parse_shape(obj: Any) -> Any:
    """Return the shape of a tensor (or nested tensors), or ``UNKNOWN_SIZE``."""
    if isinstance(obj, paddle.Tensor):
        return list(obj.shape)
    if isinstance(obj, (list, tuple)):
        shapes = [_parse_shape(o) for o in obj]
        return shapes if len(shapes) != 1 else shapes[0]
    return UNKNOWN_SIZE


class ModelSummary(Callback):
    def __init__(self, max_depth: int = 1):
        self.max_depth = max_depth

    def on_fit_start(self, trainer, model):
        self._summarize(model)

    def _summarize(self, model):
        print(self._get_summary(model))

    def _get_summary(self, model):
        layers = list(self._named_layers_depth(model, self.max_depth))
        sizes = self._collect_io_sizes(model, layers)

        lines = ["", "  | Name  | Type  | Params  | In size  | Out size  |"]
        total_params = 0
        trainable_params = 0
        for name, layer in layers:
            params = sum(p.numel() for p in layer.parameters() if not p.stop_gradient)
            non_trainable = sum(p.numel() for p in layer.parameters() if p.stop_gradient)
            total_params += params + non_trainable
            trainable_params += params
            in_size, out_size = sizes.get(name, (UNKNOWN_SIZE, UNKNOWN_SIZE))
            lines.append(
                f"  | {name} | {layer.__class__.__name__} | {params + non_trainable:,} | {in_size} | {out_size} |"
            )

        non_trainable = total_params - trainable_params
        lines.append(f"  {'─' * 50}")
        lines.append(f"  Total params: {total_params:,}")
        lines.append(f"  Trainable params: {trainable_params:,}")
        lines.append(f"  Non-trainable params: {non_trainable:,}")
        lines.append("")
        return "\n".join(lines)

    def _collect_io_sizes(self, model: Any, layers: list) -> dict:
        """Capture each layer's input/output shapes from one example forward pass.

        Requires ``model.example_input_array``; without it the sizes stay
        ``UNKNOWN_SIZE``. Forward hooks record the first pass through each layer,
        and any failure is swallowed so building a summary never crashes the run.
        """
        result = {name: (UNKNOWN_SIZE, UNKNOWN_SIZE) for name, _ in layers}
        example = getattr(model, "example_input_array", None)
        if example is None:
            return result

        captured: dict[str, tuple] = {}
        handles = []

        def make_hook(layer_name: str):
            def hook(_layer: Any, inp: Any, out: Any) -> None:
                if layer_name in captured:
                    return
                parsed_in = inp[0] if isinstance(inp, tuple) and len(inp) == 1 else inp
                captured[layer_name] = (_parse_shape(parsed_in), _parse_shape(out))

            return hook

        for name, layer in layers:
            handles.append(layer.register_forward_post_hook(make_hook(name)))

        was_training = model.training
        try:
            model.eval()
            example = self._example_to_model_device(model, example)
            with paddle.no_grad():
                self._run_example(model, example)
        except Exception:
            # Leave any uncaptured sizes as UNKNOWN_SIZE.
            pass
        finally:
            for h in handles:
                h.remove()
            if was_training:
                model.train()

        result.update(captured)
        return result

    @staticmethod
    def _example_to_model_device(model: Any, example: Any) -> Any:
        """Best-effort move of the example input to the model's parameter device."""
        try:
            place = next(iter(model.parameters())).place
        except StopIteration:
            return example

        def move(obj: Any) -> Any:
            if isinstance(obj, paddle.Tensor):
                return obj.to(place)
            if isinstance(obj, (list, tuple)):
                return type(obj)(move(o) for o in obj)
            if isinstance(obj, dict):
                return {k: move(v) for k, v in obj.items()}
            return obj

        return move(example)

    @staticmethod
    def _run_example(model: Any, example: Any) -> None:
        if isinstance(example, dict):
            model(**example)
        elif isinstance(example, (list, tuple)):
            model(*example)
        else:
            model(example)

    @staticmethod
    def _named_layers_depth(model, max_depth: int, prefix: str = "", current_depth: int = 0):
        """Yield (name, layer) respecting max_depth (ocean-compatible)."""
        if current_depth > max_depth:
            return
        for name, child in model.named_children():
            full = f"{prefix}.{name}" if prefix else name
            yield full, child
            yield from ModelSummary._named_layers_depth(child, max_depth, full, current_depth + 1)
