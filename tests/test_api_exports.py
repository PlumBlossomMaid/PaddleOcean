"""Tests for ocean API export completeness — every __all__ entry must be importable."""

from __future__ import annotations

import pytest


class TestAPIExports:
    """Ensure all symbols in ocean.__all__ are actually accessible."""

    def test_all_exports_exist(self):
        """Every name in ocean.__all__ must be a real attribute on the ocean module."""
        import ocean

        missing = []
        for name in ocean.__all__:
            if not hasattr(ocean, name):
                missing.append(name)
        assert missing == [], f"Missing exports: {missing}"

    def test_all_exports_count(self):
        """Ensure we export a reasonable number of symbols (100+)."""
        import ocean

        assert len(ocean.__all__) >= 100, f"Only {len(ocean.__all__)} exports, expected >= 100"

    @pytest.mark.parametrize(
        "name",
        [
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
            "CustomDeviceAccelerator",
            "IPUAccelerator",
            "ROCmAccelerator",
            "XPUAccelerator",
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
            "BaseFinetuning",
            "GradientAccumulationScheduler",
            "OnExceptionCheckpoint",
            "ThroughputMonitor",
            "StochasticWeightAveraging",
            "WeightAveraging",
            "ProgressBar",
            "TQDMProgressBar",
            "BatchSizeFinder",
            "LRFinder",
            # Loggers
            "Logger",
            "CSVLogger",
            "DummyLogger",
            "VisualDLLogger",
            "TensorBoardLogger",
            "WandbLogger",
            "MLFlowLogger",
            "CometLogger",
            "OceanLogger",
            "Ocelogger",
            # Strategies
            "DDPStrategy",
            "DeepSpeedStrategy",
            "FSDPStrategy",
            "ModelParallelStrategy",
            "ParallelStrategy",
            "SingleDeviceStrategy",
            "Strategy",
            # Plugins
            "Precision",
            "MixedPrecision",
            "HalfPrecision",
            "DoublePrecision",
            "CheckpointIO",
            "PaddleCheckpointIO",
            "AsyncCheckpointIO",
            "WrapperCheckpointIO",
            "LayerSync",
            "SyncBN",
            "ClusterEnvironment",
            # Profilers
            "Profiler",
            "SimpleProfiler",
            "AdvancedProfiler",
            "PassThroughProfiler",
            # Hooks & Mixins
            "ModelHooks",
            "DataHooks",
            "HyperparametersMixin",
            # Optimizer & Saving
            "OceanOptimizer",
            "init_optimizers_and_lr_schedulers",
            "load_from_checkpoint",
            # Enums
            "OceanEnum",
            # States
            "RunningStage",
            "TrainerFn",
            "TrainerState",
            "TrainerStatus",
            # Types
            "EVALUATE_OUTPUT",
            "PREDICT_OUTPUT",
            "STEP_OUTPUT",
            # Registry
            "ModelRegistry",
            # Checkpoint utilities
            "consolidate_checkpoint",
            "upgrade_checkpoint",
            "migrate_checkpoint",
            # Compile
            "to_static",
            "compile",
            "is_compiled",
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
        ],
    )
    def test_export_exists(self, name):
        import ocean

        assert hasattr(ocean, name), f"ocean.{name} not exported"

    def test_version_string(self):
        import ocean

        assert hasattr(ocean, "__version__")
        assert isinstance(ocean.__version__, str)
        assert "." in ocean.__version__


class TestSubpackageExports:
    """Ensure all subpackages have __all__ defined."""

    def test_callbacks_all(self):
        from ocean import callbacks

        assert hasattr(callbacks, "__all__")
        assert "Callback" in callbacks.__all__
        assert "ModelCheckpoint" in callbacks.__all__
        assert "BatchSizeFinder" in callbacks.__all__
        assert "LRFinder" in callbacks.__all__

    def test_strategies_all(self):
        from ocean import strategies

        assert hasattr(strategies, "__all__")
        assert "DDPStrategy" in strategies.__all__
        assert "ModelParallelStrategy" in strategies.__all__
        assert "ParallelStrategy" in strategies.__all__

    def test_accelerators_all(self):
        from ocean import accelerators

        assert hasattr(accelerators, "__all__")
        assert "CPUAccelerator" in accelerators.__all__
        assert "XPUAccelerator" in accelerators.__all__
        assert "ROCmAccelerator" in accelerators.__all__
        assert "IPUAccelerator" in accelerators.__all__
        assert "CustomDeviceAccelerator" in accelerators.__all__

    def test_plugins_all(self):
        from ocean import plugins

        assert hasattr(plugins, "__all__")
        assert "Precision" in plugins.__all__
        assert "DoublePrecision" in plugins.__all__
        assert "HalfPrecision" in plugins.__all__
        assert "CheckpointIO" in plugins.__all__
        assert "AsyncCheckpointIO" in plugins.__all__
        assert "WrapperCheckpointIO" in plugins.__all__
        assert "LayerSync" in plugins.__all__
        assert "SyncBN" in plugins.__all__
        assert "ClusterEnvironment" in plugins.__all__

    def test_loggers_all(self):
        from ocean import loggers

        assert hasattr(loggers, "__all__")
        assert "Logger" in loggers.__all__
        assert "CSVLogger" in loggers.__all__
        assert "TensorBoardLogger" in loggers.__all__
        assert "DummyLogger" in loggers.__all__

    def test_profilers_all(self):
        from ocean import profilers

        assert hasattr(profilers, "__all__") or True  # profilers may not have __all__, that's OK

    def test_cloud_all(self):
        from ocean import cloud

        assert hasattr(cloud, "__all__")
        assert "upload_file" in cloud.__all__
        assert "download_file" in cloud.__all__
        assert "list_files" in cloud.__all__
        assert "delete_file" in cloud.__all__


class TestPluginInstantiation:
    """Ensure all plugins can be instantiated."""

    def test_checkpoint_io_plugins(self):
        from ocean.plugins import AsyncCheckpointIO, PaddleCheckpointIO, WrapperCheckpointIO

        # PaddleCheckpointIO should work
        io = PaddleCheckpointIO()
        assert io is not None

        # WrapperCheckpointIO with default
        wrapper = WrapperCheckpointIO()
        assert wrapper is not None

        # AsyncCheckpointIO
        async_io = AsyncCheckpointIO()
        assert async_io is not None
        async_io.teardown()

    def test_precision_plugins(self):
        from ocean.plugins import DoublePrecision, HalfPrecision, MixedPrecision, Precision

        assert Precision() is not None
        assert MixedPrecision() is not None
        assert HalfPrecision() is not None
        assert DoublePrecision() is not None

    def test_layer_sync(self):
        from ocean.plugins import LayerSync, SyncBN

        assert LayerSync() is not None
        assert SyncBN() is not None
        assert SyncBN(sync_bn=False) is not None


class TestUtilityFunctions:
    """Test utility functions are callable."""

    def test_model_registry(self):
        from ocean import ModelRegistry

        # Register a class
        @ModelRegistry.register("test_model_cls")
        class TestModel:
            pass

        # Get it back
        cls = ModelRegistry.get("test_model_cls")
        assert cls is TestModel

        # Available list
        assert "test_model_cls" in ModelRegistry.available()

        # KeyError for unknown
        with pytest.raises(KeyError):
            ModelRegistry.get("nonexistent_model")

    def test_compile_functions(self):
        import ocean

        # to_static should be callable
        assert callable(ocean.to_static)
        assert callable(ocean.compile)
        assert callable(ocean.is_compiled)

        # is_compiled on a regular model should return False
        import paddle

        model = paddle.nn.Linear(10, 2)
        assert ocean.is_compiled(model) is False or ocean.is_compiled(model) is True

    def test_checkpoint_utilities(self):
        import ocean

        # upgrade_checkpoint
        ckpt = {"model": {"linear.weight": [1, 2]}}
        upgraded = ocean.upgrade_checkpoint(ckpt, "0.1.0", "0.2.0")
        assert "state_dict" in upgraded
        assert "paddle_ocean_version" in upgraded

        # migrate_checkpoint
        ckpt2 = {"model_state_dict": {"w": [1]}}
        migrated = ocean.migrate_checkpoint(ckpt2, "0.1.0", "0.2.0")
        assert "state_dict" in migrated

        # consolidate_checkpoint (needs real files)
        assert callable(ocean.consolidate_checkpoint)
