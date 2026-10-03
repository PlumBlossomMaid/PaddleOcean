"""ocean train — launch training from CLI."""

from __future__ import annotations

import importlib
import os
from typing import Any

import click


def _import_class(qualified_name: str):
    """Import 'module.path.ClassName' and return the class."""
    module_path, class_name = qualified_name.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _build_from_config(section: dict) -> Any:
    """Build an object from a config dict with '_target_' key.

    Example:
        {"_target_": "paddle.nn.Linear", "in_features": 28, "out_features": 10}
        → paddle.nn.Linear(28, 10)
    """
    if "_target_" not in section:
        raise ValueError(f"Config section must contain '_target_' key, got: {list(section.keys())}")
    cls = _import_class(section["_target_"])
    kwargs = {k: v for k, v in section.items() if k != "_target_"}
    return cls(**kwargs)


def _build_dataloader(config: dict, split: str = "train"):
    """Build a DataLoader from config.

    Supports two forms:
    1. Explicit class:  {"_target_": "paddle.io.DataLoader", "dataset": {...}, "batch_size": 32}
    2. Shorthand:       {"dataset": {"_target_": "my_module.MyDataset"}, "batch_size": 32}
    """
    dl_config = config.get(split, {})
    if not dl_config:
        return None

    # If _target_ is specified, build the whole thing as a class
    if "_target_" in dl_config:
        return _build_from_config(dl_config)

    # Shorthand: build dataset then wrap with DataLoader
    dataset_config = dl_config.get("dataset", {})
    if not dataset_config:
        return None

    dataset = _build_from_config(dataset_config)

    import paddle

    loader_kwargs = {
        "batch_size": dl_config.get("batch_size", 32),
        "shuffle": dl_config.get("shuffle", split == "train"),
        "num_workers": dl_config.get("num_workers", 0),
        "drop_last": dl_config.get("drop_last", split == "train"),
    }
    return paddle.io.DataLoader(dataset, **loader_kwargs)


@click.command()
@click.option("--config", "-c", default=None, help="Path to YAML config file.")
@click.option(
    "--model",
    type=str,
    default=None,
    help="Fully qualified model class, e.g. 'my_module.MyModel'.",
)
@click.option("--max-epochs", type=int, default=None, help="Max training epochs.")
@click.option("--max-steps", type=int, default=None, help="Max training steps.")
@click.option("--accelerator", type=str, default=None, help="Accelerator type (cpu/gpu/auto).")
@click.option("--devices", type=str, default=None, help="Devices to use (e.g. '1' or '0,1').")
@click.option("--precision", type=str, default=None, help="Precision mode (32/16-mixed/bf16-mixed).")
@click.option("--train-loader", type=str, default=None, help="Fully qualified DataLoader builder function.")
@click.option("--val-loader", type=str, default=None, help="Fully qualified DataLoader builder function.")
@click.option("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from.")
@click.option("--log-dir", type=str, default=None, help="Directory for logs and checkpoints.")
def train(
    config, model, max_epochs, max_steps, accelerator, devices, precision, train_loader, val_loader, checkpoint, log_dir
):
    """Start model training from CLI or config file.

    Examples:

        # Train from CLI with explicit model class
        ocean train --model my_model.MyModel --max-epochs 100 --accelerator gpu

        # Train from YAML config
        ocean train --config config.yaml

        # Train with checkpoint resume
        ocean train --config config.yaml --checkpoint checkpoints/epoch-10.pdparams

    Config file format (YAML)::

        model:
          _target_: my_model.MyModel
          hidden_size: 256

        trainer:
          max_epochs: 100
          accelerator: gpu
          precision: 16-mixed

        train:
          dataset:
            _target_: my_dataset.TrainDataset
            root: ./data
          batch_size: 32
          shuffle: true

        val:
          dataset:
            _target_: my_dataset.ValDataset
            root: ./data
          batch_size: 64
          shuffle: false

        callbacks:
          - _target_: ocean.callbacks.ModelCheckpoint
            dirpath: ./checkpoints
            save_top_k: 3
          - _target_: ocean.callbacks.EarlyStopping
            monitor: val_loss
            patience: 5
    """
    from ocean.trainer import Trainer

    trainer_kwargs = {}
    model_instance = None
    train_dl = None
    val_dl = None
    callbacks = []

    # Load YAML config if provided
    if config:
        if not os.path.exists(config):
            click.echo(f"Error: config file '{config}' not found.", err=True)
            return
        try:
            import yaml

            with open(config) as f:
                cfg = yaml.safe_load(f)
        except ImportError:
            click.echo("Error: pyyaml is required for config files. Install with: pip install pyyaml", err=True)
            return

        # Build model from config
        if "model" in cfg and not model:
            model_instance = _build_from_config(cfg["model"])

        # Trainer config
        for k, v in cfg.get("trainer", {}).items():
            if k not in trainer_kwargs or trainer_kwargs[k] is None:
                trainer_kwargs[k] = v

        # Build dataloaders from config
        if "train" in cfg and not train_loader:
            train_dl = _build_dataloader(cfg, "train")
        if "val" in cfg and not val_loader:
            val_dl = _build_dataloader(cfg, "val")

        # Build callbacks from config
        for cb_cfg in cfg.get("callbacks", []):
            callbacks.append(_build_from_config(cb_cfg))

    # CLI overrides
    if max_epochs is not None:
        trainer_kwargs["max_epochs"] = max_epochs
    if max_steps is not None:
        trainer_kwargs["max_steps"] = max_steps
    if accelerator is not None:
        trainer_kwargs["accelerator"] = accelerator
    if devices is not None:
        trainer_kwargs["devices"] = devices
    if precision is not None:
        trainer_kwargs["precision"] = precision

    # Build model from --model flag
    if model and model_instance is None:
        model_cls = _import_class(model)
        model_instance = model_cls()

    # Build dataloaders from --train-loader/--val-loader flags
    if train_loader and train_dl is None:
        builder = _import_class(train_loader)
        train_dl = builder()
    if val_loader and val_dl is None:
        builder = _import_class(val_loader)
        val_dl = builder()

    # Add callbacks to trainer
    if callbacks:
        trainer_kwargs["callbacks"] = callbacks

    # Set log directory (used by loggers and checkpoints)
    if log_dir:
        if "default_root_dir" not in trainer_kwargs:
            trainer_kwargs["default_root_dir"] = log_dir

    if model_instance is None:
        click.echo("Error: no model specified. Use --model or provide 'model' in config.", err=True)
        return

    trainer = Trainer(**trainer_kwargs)

    # Resume from checkpoint
    if checkpoint:
        click.echo(f"Resuming from {checkpoint}")
        # The Trainer handles checkpoint loading internally
        trainer._checkpoint_path = checkpoint

    click.echo(f"Training {type(model_instance).__name__} for {trainer_kwargs.get('max_epochs', '?')} epochs")
    trainer.fit(model_instance, train_dataloaders=train_dl, val_dataloaders=val_dl)
    click.echo("✅ Training complete")
