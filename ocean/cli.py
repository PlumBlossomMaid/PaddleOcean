"""CLI for launching paddleOcean training from command line.

Usage::
    python -m ocean.cli --model MyModel --trainer.max_epochs 10

Inspired by Lightning CLI.
"""

import argparse
import importlib
import inspect
import sys
from typing import Any, Optional


class OceanArgumentParser(argparse.ArgumentParser):
    """Argument parser that auto-discovers Trainer and Model parameters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._add_trainer_args()
        self._add_model_args()

    def _add_trainer_args(self) -> None:
        """Add Trainer __init__ parameters as CLI arguments."""
        from ocean.trainer import Trainer
        sig = inspect.signature(Trainer.__init__)
        for name, param in sig.parameters.items():
            if name == "self" or name == "kwargs":
                continue
            default = param.default
            if default is inspect.Parameter.empty:
                self.add_argument(f"--trainer.{name}", required=True)
            elif default is not None:
                arg_type = type(default) if default is not None else str
                self.add_argument(f"--trainer.{name}", default=default, type=arg_type)

    def _add_model_args(self) -> None:
        self.add_argument("--model", type=str, required=False,
                          help="Fully qualified model class name, e.g. 'my_module.MyModel'")
        self.add_argument("--model_config", type=str, default=None,
                          help="Path to model config file")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = OceanArgumentParser(description="paddleOcean CLI")
    return parser.parse_args()


def main() -> None:
    """Main entry point for CLI."""
    args = parse_args()

    # Import model class
    model = None
    if args.model:
        module_path, class_name = args.model.rsplit(".", 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        model = model_class()

    # Build trainer
    from ocean.trainer import Trainer
    trainer_kwargs = {}
    for key, value in vars(args).items():
        if key.startswith("trainer."):
            param_name = key[len("trainer."):]
            trainer_kwargs[param_name] = value

    trainer = Trainer(**trainer_kwargs)

    # Run
    if model is not None:
        trainer.fit(model)


if __name__ == "__main__":
    main()
