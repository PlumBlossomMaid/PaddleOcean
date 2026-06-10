"""paddleOcean - A high-level PaddlePaddle framework inspired by PyTorch Lightning.

Usage::

    import ocean

    # Use ocean's own framework
    model = ocean.Model(...)
    trainer = ocean.Trainer(...)

    # Use ALL of PaddlePaddle's API without importing paddle
    x = ocean.randn([3, 4])              # → paddle.randn
    layer = ocean.nn.Linear(10, 2)       # → paddle.nn.Linear
    loss = ocean.nn.functional.cross_entropy(...)  # → paddle.nn.functional
    opt = ocean.optimizer.Adam(...)       # → paddle.optimizer.Adam

    # Version-gated APIs work across Paddle 2.4~3.3
    y = ocean.repeat_interleave(x, 3)    # 2.5+ native, older fallback
"""

import importlib
import sys
from types import ModuleType
from typing import Any

import paddle as _paddle

# ====================================================================
# Core ocean framework components
# ====================================================================
from ocean.model import Model
from ocean.trainer import Trainer
from ocean.datamodule import DataModule
from ocean.gear import Gear

from ocean.accelerators import Accelerator, CPUAccelerator, CUDAAccelerator, GPUAccelerator

from ocean.callbacks import (
    Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor,
    Timer, ModelSummary, RichModelSummary, DeviceStatsMonitor, LambdaCallback,
    PredictionWriter, BackboneFinetuning, GradientAccumulationScheduler,
    OnExceptionCheckpoint, ThroughputMonitor, StochasticWeightAveraging,
    WeightAveraging, ProgressBar, TQDMProgressBar,
)

from ocean.loggers import Logger, CSVLogger, VisualDLLogger, WandbLogger, MLFlowLogger, CometLogger

from ocean.loops import _Loop, _FitLoop, _TrainingEpochLoop, _EvaluationLoop, _PredictionLoop
from ocean.strategies import Strategy, SingleDeviceStrategy
from ocean.plugins import Precision

from ocean.trainer.states import TrainerState, TrainerStatus, TrainerFn, RunningStage
from ocean.trainer.call import _call_callback_hooks, _call_lightning_module_hook
from ocean.trainer.connectors import _DataConnector, _LoggerConnector, _CallbackConnector, _CheckpointConnector

from ocean.core.hooks import ModelHooks, DataHooks
from ocean.core.mixins import HyperparametersMixin
from ocean.core.optimizer import OceanOptimizer, init_optimizers_and_lr_schedulers
from ocean.core.saving import load_from_checkpoint

from ocean.utils.types import STEP_OUTPUT, EVALUATE_OUTPUT, PREDICT_OUTPUT

# ====================================================================
# Compat-wrapped APIs
# ====================================================================
from ocean._compat.tensor import (
    repeat_interleave, index_add, scatter_along_axis, scatter_nd,
    take_along_axis, put_along_axis,
    masked_fill, masked_select,
    sort, argsort, unique, nonzero,
    logsumexp, lgamma,
)

# ====================================================================
# Version info
# ====================================================================
from ocean._compat.version import Version, PADDLE_VERSION, version_gte, version_lt

__version__ = "0.1.0"

# ====================================================================
# __all__
# ====================================================================
__all__ = [
    # Core
    "Model", "Trainer", "DataModule", "Gear",
    # Accelerators
    "Accelerator", "CPUAccelerator", "CUDAAccelerator", "GPUAccelerator",
    # Callbacks
    "Callback", "ModelCheckpoint", "EarlyStopping", "LearningRateMonitor",
    "Timer", "ModelSummary", "RichModelSummary", "DeviceStatsMonitor",
    "LambdaCallback", "PredictionWriter", "BackboneFinetuning",
    "GradientAccumulationScheduler", "OnExceptionCheckpoint",
    "ThroughputMonitor", "StochasticWeightAveraging", "WeightAveraging",
    "ProgressBar", "TQDMProgressBar",
    # Loggers
    "Logger", "CSVLogger", "VisualDLLogger", "WandbLogger", "MLFlowLogger", "CometLogger",
    # Compat APIs
    "repeat_interleave", "index_add", "scatter_along_axis", "scatter_nd",
    "take_along_axis", "put_along_axis", "masked_fill", "masked_select",
    "sort", "argsort", "unique", "nonzero", "logsumexp", "lgamma",
    # Version
    "Version", "PADDLE_VERSION", "version_gte", "version_lt",
]


# ====================================================================
# Dynamic paddle proxy: any attribute not defined above falls through to paddle
# ====================================================================

class _PaddleProxy(ModuleType):
    """Module proxy that resolves unknown attributes to paddle equivalents.

    This lets users write ``ocean.randn(...)``, ``ocean.nn.Linear(...)``,
    ``ocean.optimizer.Adam(...)`` without ever importing paddle directly.
    """

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") and name != "__all__":
            raise AttributeError(f"module 'ocean' has no attribute {name!r}")

        if name in self.__dict__:
            return self.__dict__[name]

        # Check _compat first (version-gated APIs)
        try:
            compat_mod = importlib.import_module(f"ocean._compat.{name}")
            setattr(self, name, compat_mod)
            return compat_mod
        except (ImportError, ModuleNotFoundError):
            pass

        # Fall through to paddle
        paddle_attr = getattr(_paddle, name, None)
        if paddle_attr is not None:
            setattr(self, name, paddle_attr)
            return paddle_attr

        # Try paddle submodule (ocean.vision → paddle.vision)
        try:
            submod = importlib.import_module(f"paddle.{name}")
            setattr(self, name, submod)
            return submod
        except (ImportError, ModuleNotFoundError):
            pass

        raise AttributeError(
            f"module 'ocean' has no attribute {name!r}. "
            f"This API may not exist in PaddlePaddle {PADDLE_VERSION}."
        )

    def __dir__(self) -> list:
        """Include both ocean's own attrs and all paddle top-level attrs."""
        ocean_attrs = set(super().__dir__())
        paddle_attrs = {x for x in dir(_paddle) if not x.startswith("_")}
        return sorted(ocean_attrs | paddle_attrs)


# Apply the proxy
_self = sys.modules[__name__]
_self.__class__ = _PaddleProxy
