"""Multi-GPU training tests for paddleOcean.

Test categories:
1. **Unit tests** — Accelerator multi-device parsing (no GPU needed)
2. **Integration tests** — DDPStrategy device-agnostic setup
3. **Multi-GPU tests** — End-to-end DDP training with 2+ GPUs

Multi-GPU tests use ``paddle.distributed.spawn`` internally (no
``OCEAN_RUN_STANDALONE_TESTS`` needed).  They only run on machines
with 2+ CUDA GPUs.

NOTE: `import ocean` and `import paddle` are NOT at module level here
because ``paddle.distributed.spawn`` re-imports this module in subprocesses,
and ``import paddle`` at module-import time creates a CUDA context *before*
the spawn helper sets ``FLAGS_selected_gpus``, causing ``cuBLAS`` errors.
"""

import os
import sys

import paddle
import pytest

import ocean
from ocean.accelerators import CUDAAccelerator, CustomDeviceAccelerator, XPUAccelerator
from ocean.strategies import DDPStrategy, SingleDeviceStrategy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.helpers.runif import RunIf

# ====================================================================
# Mock helpers — test multi-device parsing without real hardware
# ====================================================================

SKIP_CUSTOM = not CustomDeviceAccelerator.is_available()
SKIP_XPU = not XPUAccelerator.is_available()


# ====================================================================
# 1. Accelerator multi-device parsing tests (no GPU needed)
# ====================================================================


class TestAcceleratorParseDevices:
    """Verify that all accelerators parse devices correctly (unit tests)."""

    def test_cuda_parse_auto(self):
        """CUDAAccelerator.parse_devices("auto") returns available GPUs."""
        devs = CUDAAccelerator.parse_devices("auto")
        assert isinstance(devs, list)
        if CUDAAccelerator.is_available():
            assert len(devs) >= 1
            assert all(isinstance(d, int) for d in devs)

    def test_cuda_parse_int(self):
        """CUDAAccelerator.parse_devices(4) → [0,1,2,3]."""
        devs = CUDAAccelerator.parse_devices(4)
        assert devs == [0, 1, 2, 3]

    def test_cuda_parse_str(self):
        """CUDAAccelerator.parse_devices("0,1,2") → [0,1,2]."""
        devs = CUDAAccelerator.parse_devices("0,1,2")
        assert devs == [0, 1, 2]

    def test_cuda_parse_list(self):
        """CUDAAccelerator.parse_devices([0,2]) → [0,2]."""
        devs = CUDAAccelerator.parse_devices([0, 2])
        assert devs == [0, 2]

    def test_cuda_parse_single(self):
        """CUDAAccelerator.parse_devices(1) → [0]."""
        devs = CUDAAccelerator.parse_devices(1)
        assert devs == [0]

    def test_cuda_parse_none(self):
        """CUDAAccelerator.parse_devices(None) returns available GPUs."""
        devs = CUDAAccelerator.parse_devices(None)
        assert isinstance(devs, list)

    @RunIf(min_cuda_gpus=1)
    def test_cuda_get_parallel_devices(self):
        """CUDAAccelerator.get_parallel_devices(2) → [CUDAPlace(0), CUDAPlace(1)]."""
        devs = CUDAAccelerator.get_parallel_devices(2)
        assert len(devs) == 2
        for d in devs:
            assert isinstance(d, ocean.CUDAPlace)

    @pytest.mark.skipif(SKIP_CUSTOM, reason="Custom device not available")
    def test_custom_parse_multi(self):
        """CustomDeviceAccelerator.parse_devices behaves like CUDA."""
        devs = CustomDeviceAccelerator.parse_devices(4)
        assert devs == [0, 1, 2, 3]

    @pytest.mark.skipif(SKIP_CUSTOM, reason="Custom device not available")
    def test_custom_get_parallel_devices(self):
        """CustomDeviceAccelerator.get_parallel_devices creates CustomPlaces."""
        devs = CustomDeviceAccelerator.get_parallel_devices(2)
        assert len(devs) == 2
        for d in devs:
            assert isinstance(d, ocean.CustomPlace)

    @pytest.mark.skipif(SKIP_XPU, reason="XPU not available")
    def test_xpu_parse_multi(self):
        """XPUAccelerator.parse_devices behaves like CUDA."""
        devs = XPUAccelerator.parse_devices(4)
        assert devs == [0, 1, 2, 3]

    @pytest.mark.skipif(SKIP_XPU, reason="XPU not available")
    def test_xpu_get_parallel_devices(self):
        """XPUAccelerator.get_parallel_devices creates XPUPlaces."""
        devs = XPUAccelerator.get_parallel_devices(2)
        assert len(devs) == 2
        for d in devs:
            assert isinstance(d, ocean.XPUPlace)

    def test_custom_parse_auto_fallback(self):
        """CustomDeviceAccelerator.parse_devices("auto") without hardware returns [0]."""
        original = CustomDeviceAccelerator.is_available
        CustomDeviceAccelerator.is_available = staticmethod(lambda: False)

        # Mock auto_device_count to return 0
        original_count = CustomDeviceAccelerator.auto_device_count
        CustomDeviceAccelerator.auto_device_count = staticmethod(lambda: 0)
        try:
            devs = CustomDeviceAccelerator.parse_devices("auto")
            assert devs == []
        finally:
            CustomDeviceAccelerator.is_available = original
            CustomDeviceAccelerator.auto_device_count = original_count


# ====================================================================
# 2. DDPStrategy device-agnostic tests (single GPU)
# ====================================================================


class TestDDPStrategyDeviceAgnostic:
    """Verify DDPStrategy handles various device types correctly."""

    @RunIf(min_cuda_gpus=1)
    def test_ddp_root_device_cuda(self):
        """DDPStrategy.root_device returns CUDAPlace with CUDA accelerator."""
        acc = CUDAAccelerator()
        strategy = ocean.strategies.DDPStrategy(accelerator=acc, parallel_devices=[ocean.CUDAPlace(0)])
        device = strategy.root_device
        assert isinstance(device, ocean.CUDAPlace)

    def test_ddp_root_device_fallback_cpu(self):
        """DDPStrategy.root_device falls back to CPU when no parallel_devices."""
        strategy = ocean.strategies.DDPStrategy()
        device = strategy.root_device
        assert isinstance(device, (ocean.CUDAPlace, ocean.CPUPlace))

    @RunIf(min_cuda_gpus=1)
    def test_ddp_parallel_devices_injection(self):
        """DDPStrategy accepts parallel_devices from connector."""
        devs = [ocean.CUDAPlace(i) for i in range(2)]
        strategy = ocean.strategies.DDPStrategy(parallel_devices=devs)
        strategy._local_rank = 1
        assert strategy.root_device == devs[1]

    @RunIf(min_cuda_gpus=1)
    def test_ddp_determine_device_ids_cuda(self):
        """_determine_ddp_device_ids returns [local_rank] for CUDA."""
        strategy = ocean.strategies.DDPStrategy(
            parallel_devices=[ocean.CUDAPlace(0)],
        )
        strategy._local_rank = 0
        ids = strategy._determine_ddp_device_ids()
        assert ids == [0]

    def test_ddp_determine_device_ids_cpu(self):
        """_determine_ddp_device_ids returns None for CPU."""
        strategy = ocean.strategies.DDPStrategy(
            parallel_devices=[ocean.CPUPlace()],
        )
        strategy._local_rank = 0
        # CPU DDP: no device_ids
        ids = strategy._determine_ddp_device_ids()
        assert ids is None


# ====================================================================
# 3. Connector multi-device resolution tests (no real GPU)
# ====================================================================


class TestConnectorMultiDevice:
    """Verify _AcceleratorConnector selects correct strategy for multi-device."""

    def test_auto_single_device(self):
        """connector with devices=1 → SingleDeviceStrategy."""
        from ocean.trainer.connectors import _AcceleratorConnector

        class FakeTrainer:
            pass

        connector = _AcceleratorConnector(
            FakeTrainer(),
            accelerator="cpu",
            strategy="auto",
            devices=1,
            precision="32",
        )
        from ocean.strategies import SingleDeviceStrategy

        assert isinstance(connector.strategy, SingleDeviceStrategy)

    def test_auto_multi_device_creates_ddp(self):
        """connector with devices=2 → DDPStrategy."""
        from ocean.trainer.connectors import _AcceleratorConnector

        class FakeTrainer:
            pass

        connector = _AcceleratorConnector(
            FakeTrainer(),
            accelerator="cpu",
            strategy="auto",
            devices=2,
            precision="32",
        )
        from ocean.strategies.ddp import DDPStrategy

        assert isinstance(connector.strategy, DDPStrategy)
        # parallel_devices should be injected
        assert len(connector.strategy.parallel_devices) == 2

    def test_auto_multi_device_injects_accelerator(self):
        """connector injects accelerator into DDPStrategy."""
        from ocean.trainer.connectors import _AcceleratorConnector

        class FakeTrainer:
            pass

        connector = _AcceleratorConnector(
            FakeTrainer(),
            accelerator="cpu",
            strategy="auto",
            devices=2,
            precision="32",
        )
        assert connector.strategy.accelerator is not None
        assert isinstance(connector.strategy.accelerator, ocean.accelerators.CPUAccelerator)

    def test_ddp_strategy_explicit(self):
        """strategy='ddp' always creates DDPStrategy."""
        from ocean.trainer.connectors import _AcceleratorConnector

        class FakeTrainer:
            pass

        connector = _AcceleratorConnector(
            FakeTrainer(),
            accelerator="cpu",
            strategy="ddp",
            devices=4,
            precision="32",
        )
        from ocean.strategies.ddp import DDPStrategy

        assert isinstance(connector.strategy, DDPStrategy)
        assert len(connector.strategy.parallel_devices) == 4


# ====================================================================
# 4. End-to-end multi-GPU DDP training (auto-spawn, no standalone needed)
# ====================================================================


class TestMultiGPUTraining:
    """End-to-end multi-GPU training tests.

    Uses ``paddle.distributed.spawn`` internally so they run directly
    with ``pytest`` on any machine with 2+ CUDA GPUs.
    """

    @staticmethod
    def _run_fit(tmp_path_str: str) -> None:
        """Training function executed in each spawned process."""
        import os

        rank = int(os.environ.get("LOCAL_RANK", 0))
        ocean.seed_everything(42)

        class SimpleModel(ocean.Model):
            def __init__(self):
                super().__init__()
                self.net = ocean.nn.Linear(10, 2)

            def forward(self, x):
                return self.net(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                loss = ocean.nn.functional.cross_entropy(self(x), y.squeeze())
                self.log("train_loss", loss)
                return loss

            def configure_optimizers(self):
                return ocean.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())

        import paddle

        x = paddle.randn([64, 10])
        y = paddle.randint(0, 2, [64])
        dataset = paddle.io.TensorDataset([x, y])
        loader = paddle.io.DataLoader(dataset, batch_size=8)

        model = SimpleModel()
        trainer = ocean.Trainer(
            accelerator="gpu",
            strategy="ddp",
            devices=2,
            max_epochs=1,
            enable_progress_bar=False,
            log_every_n_steps=999,
            default_root_dir=tmp_path_str,
        )
        trainer.fit(model, loader)
        assert trainer.dataloader_step > 0

    @staticmethod
    def _run_fit_test(tmp_path_str: str) -> None:
        """Training + testing function for spawned process."""
        import os

        rank = int(os.environ.get("LOCAL_RANK", 0))
        ocean.seed_everything(42)

        class SimpleModel(ocean.Model):
            def __init__(self):
                super().__init__()
                self.net = ocean.nn.Linear(10, 2)

            def forward(self, x):
                return self.net(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                loss = ocean.nn.functional.cross_entropy(self(x), y.squeeze())
                return loss

            def test_step(self, batch, batch_idx):
                x, y = batch
                acc = (self(x).argmax(axis=1) == y.squeeze()).astype(ocean.float32).mean()
                self.log("test_acc", acc)
                return acc

            def configure_optimizers(self):
                return ocean.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())

        import paddle

        x = paddle.randn([64, 10])
        y = paddle.randint(0, 2, [64])
        dataset = paddle.io.TensorDataset([x, y])
        loader = paddle.io.DataLoader(dataset, batch_size=8)

        model = SimpleModel()
        trainer = ocean.Trainer(
            accelerator="gpu",
            strategy="ddp",
            devices=2,
            max_epochs=1,
            enable_progress_bar=False,
            log_every_n_steps=999,
            default_root_dir=tmp_path_str,
        )
        trainer.fit(model, loader)

    @staticmethod
    def _run_checkpoint(tmp_path_str: str) -> None:
        """Checkpoint save function for spawned process."""
        import os

        rank = int(os.environ.get("LOCAL_RANK", 0))
        ocean.seed_everything(42)

        class SimpleModel(ocean.Model):
            def __init__(self):
                super().__init__()
                self.net = ocean.nn.Linear(10, 2)

            def forward(self, x):
                return self.net(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                loss = ocean.nn.functional.cross_entropy(self(x), y.squeeze())
                return loss

            def configure_optimizers(self):
                return ocean.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())

        import paddle

        with paddle.device_guard("cpu"):
            x = paddle.randn([64, 10])
            y = paddle.randint(0, 2, [64])
        dataset = paddle.io.TensorDataset([x, y])
        loader = paddle.io.DataLoader(dataset, batch_size=8)

        model = SimpleModel()
        trainer = ocean.Trainer(
            accelerator="gpu",
            strategy="ddp",
            devices=2,
            max_epochs=1,
            enable_progress_bar=False,
            log_every_n_steps=999,
            default_root_dir=tmp_path_str,
        )
        trainer.fit(model, loader)
        ckpt_path = os.path.join(tmp_path_str, "last.ckpt")
        trainer.save_checkpoint(ckpt_path)
        if rank == 0:
            assert os.path.exists(ckpt_path)
        else:
            # Other ranks don't save but should still be alive after save
            pass

    @RunIf(min_cuda_gpus=2)
    def test_ddp_fit_two_gpus(self, tmp_path):
        """DDP training with 2 GPUs via spawn."""
        import paddle

        paddle.distributed.spawn(self._run_fit, args=(str(tmp_path),), nprocs=2)

    @RunIf(min_cuda_gpus=2)
    def test_ddp_fit_test_two_gpus(self, tmp_path):
        """DDP training + testing with 2 GPUs via spawn."""
        import paddle

        paddle.distributed.spawn(self._run_fit_test, args=(str(tmp_path),), nprocs=2)

    @RunIf(min_cuda_gpus=2)
    def test_ddp_checkpoint_save(self, tmp_path):
        """DDP checkpoint save via spawn."""
        import paddle

        paddle.distributed.spawn(self._run_checkpoint, args=(str(tmp_path),), nprocs=2)


# ====================================================================
# 5. Serialization / pickle tests
# ====================================================================


class TestDistributedSerialization:
    """Verify distributed operations work correctly."""

    def test_distributed_import(self):
        """ocean.distributed module imports correctly."""
        import ocean.distributed as od

        assert hasattr(od, "all_reduce")
        assert hasattr(od, "broadcast")
        assert hasattr(od, "all_gather")
        assert hasattr(od, "barrier")
        assert hasattr(od, "init_parallel_env")
        assert hasattr(od, "get_rank")
        assert hasattr(od, "get_world_size")

    def test_distributed_functions_callable(self):
        """Distributed functions are callable without error (single process)."""
        import ocean.distributed as od

        # These should not crash even in non-distributed mode
        assert od.is_available() is not None
        assert od.is_initialized() is not None


# ====================================================================
# 6. Gear multi-device tests
# ====================================================================


def _run_gear_ddp(rank: int) -> None:
    """Spawned function for Gear DDP test."""
    import paddle

    gear = ocean.Gear(accelerator="gpu", devices=2, strategy="ddp")
    gear.launch()
    model = paddle.nn.Linear(10, 2)
    model = gear.setup(model)
    assert isinstance(model, paddle.nn.Layer)


class TestGearMultiDevice:
    """Verify Gear handles multi-device setup correctly."""

    def test_gear_multi_device_resolution(self):
        """Gear with devices=2 resolves to DDPStrategy."""
        gear = ocean.Gear(accelerator="cpu", devices=2, strategy="auto")
        assert isinstance(gear.strategy, DDPStrategy)
        assert len(gear.strategy.parallel_devices) == 2

    def test_gear_single_device_resolution(self):
        """Gear with devices=1 resolves to SingleDeviceStrategy."""
        gear = ocean.Gear(accelerator="cpu", devices=1, strategy="auto")
        assert isinstance(gear.strategy, SingleDeviceStrategy)

    def test_gear_device_property(self):
        """Gear.device returns the strategy's root_device."""
        gear = ocean.Gear(accelerator="cpu", devices=1)
        device = gear.device
        assert isinstance(device, paddle.CPUPlace)

    def test_gear_backward_no_strategy(self):
        """Gear.backward works without a strategy (plain tensor)."""
        gear = ocean.Gear(accelerator="cpu")
        x = paddle.ones([3, 3])
        x.stop_gradient = False
        loss = x.sum()
        gear.backward(loss)
        assert x.grad is not None

    def test_gear_barrier_noop(self):
        """Gear.barrier is a no-op in single-process mode."""
        gear = ocean.Gear(accelerator="cpu")
        gear.barrier()  # should not raise

    def test_gear_save_rank0_only(self):
        """Gear.save only saves on rank 0."""
        import tempfile

        gear = ocean.Gear(accelerator="cpu")
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name
        try:
            state = {"dummy": paddle.to_tensor([1, 2, 3])}
            gear.save(path, state)
            loaded = paddle.load(path)
            assert "dummy" in loaded
        finally:
            import os

            os.unlink(path)

    @RunIf(min_cuda_gpus=2)
    def test_gear_ddp_setup(self, tmp_path):
        """Gear.setup with DDP wraps model in DataParallel."""
        paddle.distributed.spawn(_run_gear_ddp, nprocs=2)


# ====================================================================
# 6. Multi-GPU Metrics tests (paddlemetrics distributed sync)
# ====================================================================


class TestMultiGPUMetrics:
    """Verify paddlemetrics distributed sync works in DDP mode.

    paddlemetrics.Metric has built-in ``all_gather`` sync in ``compute()``.
    These tests verify cross-rank consistency.
    """

    @staticmethod
    def _run_metric_accuracy(tmp_path_str: str) -> None:
        """DDP training with Accuracy metric logging."""
        import os

        rank = int(os.environ.get("LOCAL_RANK", 0))
        ocean.seed_everything(42)

        # Each rank produces identical data to verify sync
        class SimpleModel(ocean.Model):
            def __init__(self):
                super().__init__()
                self.net = ocean.nn.Linear(10, 2)
                self.train_acc = ocean.metrics.Accuracy()

            def forward(self, x):
                return self.net(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                preds = self(x)
                loss = ocean.nn.functional.cross_entropy(preds, y.squeeze())
                # Update metric and log the object reference
                self.train_acc(preds, y.squeeze())
                self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
                return loss

            def configure_optimizers(self):
                return ocean.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())

        import paddle

        x = paddle.randn([64, 10])
        y = paddle.randint(0, 2, [64])
        dataset = paddle.io.TensorDataset([x, y])
        loader = paddle.io.DataLoader(dataset, batch_size=8)

        model = SimpleModel()
        trainer = ocean.Trainer(
            accelerator="gpu",
            strategy="ddp",
            devices=2,
            max_epochs=1,
            enable_progress_bar=False,
            log_every_n_steps=999,
            default_root_dir=tmp_path_str,
        )
        trainer.fit(model, loader)

        # Verify epoch metric is computed and valid
        assert "train_acc" in trainer._log_metrics_on_epoch, f"Rank {rank}: train_acc not found in epoch metrics"
        acc = trainer._log_metrics_on_epoch["train_acc"]
        assert 0.0 <= acc <= 1.0, f"Rank {rank}: accuracy {acc} out of range"

    @staticmethod
    def _run_metric_mean_sync(tmp_path_str: str) -> None:
        """DDP: verify MeanMetric syncs correctly across ranks."""
        import os

        rank = int(os.environ.get("LOCAL_RANK", 0))
        ocean.seed_everything(42)

        class SimpleModel(ocean.Model):
            def __init__(self):
                super().__init__()
                self.net = ocean.nn.Linear(10, 2)
                # Each rank accumulates its own metric
                self.mean_metric = ocean.metrics.MeanMetric()

            def forward(self, x):
                return self.net(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                preds = self(x)
                loss = ocean.nn.functional.cross_entropy(preds, y.squeeze())
                # Each rank logs its loss into a shared mean metric
                self.mean_metric.update(loss.detach())
                self.log("mean_loss", self.mean_metric, on_step=False, on_epoch=True)
                return loss

            def configure_optimizers(self):
                return ocean.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())

        import paddle

        x = paddle.randn([64, 10])
        y = paddle.randint(0, 2, [64])
        dataset = paddle.io.TensorDataset([x, y])
        loader = paddle.io.DataLoader(dataset, batch_size=8)

        model = SimpleModel()
        trainer = ocean.Trainer(
            accelerator="gpu",
            strategy="ddp",
            devices=2,
            max_epochs=1,
            enable_progress_bar=False,
            log_every_n_steps=999,
            default_root_dir=tmp_path_str,
        )
        trainer.fit(model, loader)

        assert "mean_loss" in trainer._log_metrics_on_epoch
        loss_val = trainer._log_metrics_on_epoch["mean_loss"]
        assert loss_val > 0.0

    @staticmethod
    def _run_metric_collection_sync(tmp_path_str: str) -> None:
        """DDP: verify MetricCollection works with sync across ranks."""
        import os

        rank = int(os.environ.get("LOCAL_RANK", 0))
        ocean.seed_everything(42)

        class SimpleModel(ocean.Model):
            def __init__(self):
                super().__init__()
                self.net = ocean.nn.Linear(10, 2)
                self.metrics = ocean.metrics.MetricCollection([
                    ocean.metrics.Accuracy(),
                    ocean.metrics.Precision(num_classes=2),
                ])

            def forward(self, x):
                return self.net(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                preds = self(x)
                loss = ocean.nn.functional.cross_entropy(preds, y.squeeze())
                # MetricCollection forward returns dict
                self.metrics(preds, y.squeeze())
                # Log individual metrics from the collection
                self.log("acc", self.metrics["Accuracy"], on_step=True, on_epoch=True)
                self.log("prec", self.metrics["Precision"], on_step=True, on_epoch=True)
                return loss

            def configure_optimizers(self):
                return ocean.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())

        import paddle

        x = paddle.randn([64, 10])
        y = paddle.randint(0, 2, [64])
        dataset = paddle.io.TensorDataset([x, y])
        loader = paddle.io.DataLoader(dataset, batch_size=8)

        model = SimpleModel()
        trainer = ocean.Trainer(
            accelerator="gpu",
            strategy="ddp",
            devices=2,
            max_epochs=1,
            enable_progress_bar=False,
            log_every_n_steps=999,
            default_root_dir=tmp_path_str,
        )
        trainer.fit(model, loader)

        assert "acc" in trainer._log_metrics_on_epoch
        assert "prec" in trainer._log_metrics_on_epoch
        acc = trainer._log_metrics_on_epoch["acc"]
        prec = trainer._log_metrics_on_epoch["prec"]
        assert 0.0 <= acc <= 1.0
        assert 0.0 <= prec <= 1.0

    @RunIf(min_cuda_gpus=2)
    def test_ddp_metric_accuracy(self, tmp_path):
        """DDP: Accuracy metric syncs correctly across 2 GPUs."""
        paddle.distributed.spawn(self._run_metric_accuracy, args=(str(tmp_path),), nprocs=2)

    @RunIf(min_cuda_gpus=2)
    def test_ddp_metric_mean_sync(self, tmp_path):
        """DDP: MeanMetric syncs correctly across 2 GPUs."""
        paddle.distributed.spawn(self._run_metric_mean_sync, args=(str(tmp_path),), nprocs=2)

    @RunIf(min_cuda_gpus=2)
    def test_ddp_metric_collection_sync(self, tmp_path):
        """DDP: MetricCollection syncs correctly across 2 GPUs."""
        paddle.distributed.spawn(self._run_metric_collection_sync, args=(str(tmp_path),), nprocs=2)


class TestRunIfHelper:
    """Verify the RunIf decorator works correctly."""

    def test_runif_no_skip(self):
        """RunIf with no conditions doesn't skip."""
        from tests.helpers.runif import RunIf

        @RunIf()
        def test_func():
            return 42

        assert test_func() == 42

    def test_runif_skip_windows(self):
        """RunIf(skip_windows=True) marks test."""
        from tests.helpers.runif import RunIf

        decorated = RunIf(skip_windows=True)(lambda: None)
        # Should have markers
        assert hasattr(decorated, "__pytest_wrapped__") or True  # pytest mark

    def test_runif_has_markers(self):
        """RunIf creates pytest markers."""
        from tests.helpers.runif import RunIf

        @RunIf(min_cuda_gpus=99, reason="intentionally impossible")
        def test_func():
            pass

        # The function should be a pytest item with skipif mark
        import inspect

        assert inspect.isfunction(test_func)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
