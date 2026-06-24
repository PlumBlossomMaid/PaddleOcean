"""paddleOcean - A high-level PaddlePaddle framework inspired by paddleOcean.

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
import warnings
from types import ModuleType
from typing import Any

# Suppress setuptools internal deprecation warning (editable install noise)
warnings.filterwarnings("ignore", message=".*_get_vc_env.*")

import paddle as _paddle

# ====================================================================
# Suppress verbose PaddlePaddle warnings (noisy per-batch info)
# ====================================================================
warnings.filterwarnings(
    "ignore",
    message="When training, we now always track global mean and variance",
    category=UserWarning,
    module="paddle.nn.layer.norm",
)

# Suppress PIR "no place interface" false positive (CINN static graph mode)
warnings.filterwarnings(
    "ignore",
    message=".*do not have 'place' interface for pir graph mode.*",
    category=UserWarning,
)

# Suppress PIR "blocking not supported" false positive (static graph .cpu())
warnings.filterwarnings(
    "ignore",
    message="blocking is not supported.*",
    category=UserWarning,
)

# ====================================================================
# Compat-wrapped APIs
# ====================================================================
# ====================================================================
# Paddle SOT compatibility patch
# ====================================================================
from ocean._compat.sot import patch_sot as _patch_sot
from ocean._compat.tensor import (
    argsort,
    index_add,
    lgamma,
    logsumexp,
    masked_fill,
    masked_select,
    nonzero,
    put_along_axis,
    repeat_interleave,
    scatter_along_axis,
    scatter_nd,
    sort,
    take_along_axis,
    unique,
)

# ====================================================================
# Version info
# ====================================================================
from ocean._compat.version import PADDLE_VERSION, Version, version_gte, version_lt
from ocean.accelerators import Accelerator, CPUAccelerator, CUDAAccelerator, GPUAccelerator
from ocean.callbacks import (
    BackboneFinetuning,
    Callback,
    DeviceStatsMonitor,
    EarlyStopping,
    GradientAccumulationScheduler,
    LambdaCallback,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    OnExceptionCheckpoint,
    PredictionWriter,
    ProgressBar,
    RichModelSummary,
    StochasticWeightAveraging,
    ThroughputMonitor,
    Timer,
    TQDMProgressBar,
    WeightAveraging,
)
from ocean.core.hooks import DataHooks, ModelHooks
from ocean.core.mixins import HyperparametersMixin
from ocean.core.optimizer import OceanOptimizer, init_optimizers_and_lr_schedulers
from ocean.core.saving import load_from_checkpoint
from ocean.datamodule import DataModule
from ocean.gear import Gear
from ocean.loggers import CometLogger, CSVLogger, Logger, MLFlowLogger, VisualDLLogger, WandbLogger
from ocean.loggers.ocelogger import OceanLogger, Ocelogger
from ocean.loops import _EvaluationLoop, _FitLoop, _Loop, _PredictionLoop, _TrainingEpochLoop

# ====================================================================
# Core ocean framework components
# ====================================================================
from ocean.model import Model
from ocean.plugins import MixedPrecision, Precision
from ocean.strategies import DDPStrategy, DeepSpeedStrategy, FSDPStrategy, SingleDeviceStrategy, Strategy
from ocean.trainer import Trainer
from ocean.trainer.call import _call_callback_hooks, _call_module_hook
from ocean.trainer.connectors import _CallbackConnector, _CheckpointConnector, _DataConnector, _LoggerConnector
from ocean.trainer.states import RunningStage, TrainerFn, TrainerState, TrainerStatus
from ocean.utils.enums import OceanEnum
from ocean.utils.seed import seed_everything
from ocean.utils.types import EVALUATE_OUTPUT, PREDICT_OUTPUT, STEP_OUTPUT

_patch_sot()

__version__ = "0.1.0"

# ====================================================================
# __all__
# ====================================================================
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
    "OceanLogger",
    "Ocelogger",
    # Strategies
    "DDPStrategy",
    "DeepSpeedStrategy",
    "FSDPStrategy",
    # Plugins
    "MixedPrecision",
    # Enums
    "OceanEnum",
    # Compat APIs
    "repeat_interleave",
    "index_add",
    "scatter_along_axis",
    "scatter_nd",
    "take_along_axis",
    "put_along_axis",
    "masked_fill",
    "masked_select",
    "sort",
    "argsort",
    "unique",
    "nonzero",
    "logsumexp",
    "lgamma",
    # Version
    "Version",
    "PADDLE_VERSION",
    "version_gte",
    "version_lt",
    # Ocean utilities
    "seed_everything",
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

        # Check ocean submodules (ocean.metrics, ocean.utils, ...)
        try:
            ocean_sub = importlib.import_module(f"ocean.{name}")
            setattr(self, name, ocean_sub)
            return ocean_sub
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
            f"module 'ocean' has no attribute {name!r}. This API may not exist in PaddlePaddle {PADDLE_VERSION}."
        )

    def __dir__(self) -> list:
        """Include both ocean's own attrs and all paddle top-level attrs."""
        ocean_attrs = set(super().__dir__())
        paddle_attrs = {x for x in dir(_paddle) if not x.startswith("_")}
        return sorted(ocean_attrs | paddle_attrs)


# Apply the proxy
_self = sys.modules[__name__]
_self.__class__ = _PaddleProxy
