"""paddleOcean - A high-level PaddlePaddle framework inspired by PyTorch Lightning."""

import paddle

# Core
from ocean.model import Model
from ocean.trainer import Trainer
from ocean.datamodule import DataModule
from ocean.gear import Gear

# Accelerators
from ocean.accelerators import Accelerator, CPUAccelerator, CUDAAccelerator, GPUAccelerator

# Callbacks
from ocean.callbacks import (
    Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor,
    Timer, ModelSummary, RichModelSummary, DeviceStatsMonitor, LambdaCallback,
    PredictionWriter, BackboneFinetuning, GradientAccumulationScheduler,
    OnExceptionCheckpoint, ThroughputMonitor, StochasticWeightAveraging,
    WeightAveraging, ProgressBar, TQDMProgressBar,
)

# Loggers
from ocean.loggers import Logger, CSVLogger, VisualDLLogger, WandbLogger, MLFlowLogger, CometLogger

# Loops
from ocean.loops import _Loop, _FitLoop, _TrainingEpochLoop, _EvaluationLoop, _PredictionLoop

# Strategies
from ocean.strategies import Strategy, SingleDeviceStrategy

# Plugins
from ocean.plugins import Precision

# Trainer submodules
from ocean.trainer.states import TrainerState, TrainerStatus, TrainerFn, RunningStage
from ocean.trainer.call import _call_callback_hooks, _call_lightning_module_hook
from ocean.trainer.connectors import _DataConnector, _LoggerConnector, _CallbackConnector, _CheckpointConnector

# Core
from ocean.core.hooks import ModelHooks, DataHooks
from ocean.core.mixins import HyperparametersMixin
from ocean.core.optimizer import OceanOptimizer, init_optimizers_and_lr_schedulers
from ocean.core.saving import load_from_checkpoint

# Utilities
from ocean.utils.types import STEP_OUTPUT, EVALUATE_OUTPUT, PREDICT_OUTPUT

__version__ = "0.1.0"

__all__ = [
    # Core
    "Model",
    "Trainer",
    "DataModule",
    "Gear",
    # Accelerators
    "Accelerator",
    "CPUAccelerator",
    "CUDAAccelerator",
    "GPUAccelerator",
    # Callbacks
    "Callback",
    "ModelCheckpoint",
    "EarlyStopping",
    "LearningRateMonitor",
    "Timer",
    "ModelSummary",
    "RichModelSummary",
    "DeviceStatsMonitor",
    "LambdaCallback",
    "PredictionWriter",
    "BackboneFinetuning",
    "GradientAccumulationScheduler",
    "OnExceptionCheckpoint",
    "ThroughputMonitor",
    "StochasticWeightAveraging",
    "WeightAveraging",
    "ProgressBar",
    "TQDMProgressBar",
    # Loggers
    "Logger",
    "CSVLogger",
    "VisualDLLogger",
    "WandbLogger",
    "MLFlowLogger",
    "CometLogger",
    # Loops
    "_Loop",
    "_FitLoop",
    "_TrainingEpochLoop",
    "_EvaluationLoop",
    "_PredictionLoop",
    # Strategies
    "Strategy",
    "SingleDeviceStrategy",
    # Plugins
    "Precision",
    # Trainer internals
    "TrainerState",
    "TrainerStatus",
    "TrainerFn",
    "RunningStage",
    "_call_callback_hooks",
    "_call_lightning_module_hook",
    "_DataConnector",
    "_LoggerConnector",
    "_CallbackConnector",
    "_CheckpointConnector",
    # Core
    "ModelHooks",
    "DataHooks",
    "HyperparametersMixin",
    "OceanOptimizer",
    "init_optimizers_and_lr_schedulers",
    "load_from_checkpoint",
    # Utilities
    "STEP_OUTPUT",
    "EVALUATE_OUTPUT",
    "PREDICT_OUTPUT",
]
