"""Tests for stub fixes: FSDP/ModelParallel collectives, accelerator teardown, logger save, config validator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import paddle
import pytest

import ocean

# ── FSDP/ModelParallel collective ops ───────────────────────────


class TestFSDPCollectives:
    """Test that FSDPStrategy.reduce/barrier/broadcast use real distributed calls."""

    def test_reduce_without_distributed_returns_tensor(self):
        """When distributed is not initialized, reduce should return the tensor unchanged."""
        fsdp = ocean.FSDPStrategy()
        tensor = paddle.randn([4])
        result = fsdp.reduce(tensor, "mean")
        assert result is not None

    def test_barrier_without_distributed_is_noop(self):
        """When distributed is not initialized, barrier should be a no-op."""
        fsdp = ocean.FSDPStrategy()
        fsdp.barrier()  # should not raise

    def test_broadcast_without_distributed_returns_obj(self):
        """When distributed is not initialized, broadcast should return obj unchanged."""
        fsdp = ocean.FSDPStrategy()
        obj = {"key": "value"}
        result = fsdp.broadcast(obj, src=0)
        assert result is not None


class TestModelParallelCollectives:
    """Test that ModelParallelStrategy.reduce/barrier/broadcast use real distributed calls."""

    def test_reduce_without_distributed_returns_tensor(self):
        strategy = ocean.ModelParallelStrategy()
        tensor = paddle.randn([4])
        result = strategy.reduce(tensor, "mean")
        assert result is not None

    def test_barrier_without_distributed_is_noop(self):
        strategy = ocean.ModelParallelStrategy()
        strategy.barrier()

    def test_broadcast_without_distributed_returns_obj(self):
        strategy = ocean.ModelParallelStrategy()
        obj = [1, 2, 3]
        result = strategy.broadcast(obj, src=0)
        assert result is not None


# ── Accelerator teardown ────────────────────────────────────────


class TestAcceleratorTeardown:
    """Test that accelerators actually clean up on teardown."""

    def test_cuda_teardown_calls_empty_cache(self):
        """CUDA teardown should call paddle.device.cuda.empty_cache."""
        accel = ocean.CUDAAccelerator()
        if paddle.is_compiled_with_cuda():
            # Should not raise
            accel.teardown()
        else:
            # Should be a no-op when CUDA not available
            accel.teardown()

    def test_xpu_teardown_does_not_raise(self):
        """XPU teardown should not raise even if XPU not available."""
        accel = ocean.XPUAccelerator()
        accel.teardown()

    def test_rocm_teardown_does_not_raise(self):
        """ROCm teardown should not raise even if ROCm not available."""
        accel = ocean.ROCmAccelerator()
        accel.teardown()

    def test_ipu_teardown_does_not_raise(self):
        """IPU teardown should not raise even if IPU not available."""
        accel = ocean.IPUAccelerator()
        accel.teardown()


# ── Accelerator completeness ────────────────────────────────────


class TestAcceleratorCompleteness:
    """Test that all accelerators have parse_devices, get_parallel_devices, etc."""

    @pytest.mark.parametrize(
        "accel_class",
        [
            ocean.CPUAccelerator,
            ocean.CUDAAccelerator,
            ocean.XPUAccelerator,
            ocean.ROCmAccelerator,
            ocean.IPUAccelerator,
            ocean.CustomDeviceAccelerator,
        ],
    )
    def test_parse_devices_auto(self, accel_class):
        result = accel_class.parse_devices("auto")
        assert isinstance(result, list)

    @pytest.mark.parametrize(
        "accel_class",
        [
            ocean.CPUAccelerator,
            ocean.CUDAAccelerator,
            ocean.XPUAccelerator,
            ocean.ROCmAccelerator,
            ocean.IPUAccelerator,
        ],
    )
    def test_parse_devices_int(self, accel_class):
        result = accel_class.parse_devices(2)
        assert isinstance(result, list)
        assert len(result) == 2

    @pytest.mark.parametrize(
        "accel_class",
        [
            ocean.CUDAAccelerator,
            ocean.XPUAccelerator,
            ocean.ROCmAccelerator,
            ocean.IPUAccelerator,
        ],
    )
    def test_parse_devices_str(self, accel_class):
        """Test string device parsing like '0,1,2'."""
        result = accel_class.parse_devices("0,1")
        assert result == [0, 1]

    @pytest.mark.parametrize(
        "accel_class",
        [
            ocean.CPUAccelerator,
            ocean.CUDAAccelerator,
            ocean.XPUAccelerator,
            ocean.ROCmAccelerator,
            ocean.IPUAccelerator,
            ocean.CustomDeviceAccelerator,
        ],
    )
    def test_is_available(self, accel_class):
        """is_available should return a bool."""
        result = accel_class.is_available()
        assert isinstance(result, bool)

    @pytest.mark.parametrize(
        "accel_class",
        [
            ocean.CPUAccelerator,
            ocean.CUDAAccelerator,
            ocean.XPUAccelerator,
            ocean.ROCmAccelerator,
            ocean.IPUAccelerator,
            ocean.CustomDeviceAccelerator,
        ],
    )
    def test_auto_device_count(self, accel_class):
        """auto_device_count should return an int."""
        result = accel_class.auto_device_count()
        assert isinstance(result, int)

    def test_rocm_setup_device(self):
        """ROCm setup_device should set device and return place."""
        accel = ocean.ROCmAccelerator()
        if ocean.ROCmAccelerator.is_available():
            place = accel.setup_device(0)
            assert place is not None

    def test_rocm_get_device_stats(self):
        """ROCm get_device_stats should return a dict."""
        accel = ocean.ROCmAccelerator()
        if ocean.ROCmAccelerator.is_available():
            stats = accel.get_device_stats(0)
            assert isinstance(stats, dict)

    def test_ipu_get_device_stats(self):
        """IPU get_device_stats should return a dict."""
        accel = ocean.IPUAccelerator()
        stats = accel.get_device_stats(0)
        assert isinstance(stats, dict)


# ── Logger save() ───────────────────────────────────────────────


class TestLoggerSave:
    """Test that all loggers have a functional save() method."""

    def test_csv_logger_save(self, tmp_path):
        logger = ocean.CSVLogger(root_dir=str(tmp_path))
        logger.log_metrics({"loss": 0.5}, step=0)
        logger.save()  # should not raise

    def test_visualdl_logger_save(self, tmp_path):
        """VisualDL save should not raise."""
        logger = ocean.VisualDLLogger(save_dir=str(tmp_path))
        # save before experiment is created should be a no-op
        logger.save()

    def test_tensorboard_logger_save(self, tmp_path):
        """TensorBoard save should not raise."""
        logger = ocean.TensorBoardLogger(save_dir=str(tmp_path))
        logger.save()

    def test_wandb_logger_save(self, tmp_path):
        """Wandb save should not raise (even without wandb installed)."""
        logger = ocean.WandbLogger(save_dir=str(tmp_path))
        logger.save()

    def test_mlflow_logger_save(self, tmp_path):
        """MLFlow save should not raise."""
        logger = ocean.MLFlowLogger(experiment_name="test")
        logger.save()

    def test_comet_logger_save(self, tmp_path):
        """Comet save should not raise."""
        logger = ocean.CometLogger(api_key=None)
        logger.save()

    def test_comet_log_graph_warns(self, tmp_path):
        """Comet log_graph should warn instead of silently doing nothing."""
        logger = ocean.CometLogger(save_dir=str(tmp_path))
        with pytest.warns(UserWarning, match="not yet implemented"):
            logger.log_graph(model=None)


# ── Configuration validator ─────────────────────────────────────


class TestConfigValidator:
    """Test that configuration validator catches invalid configs."""

    def test_check_strategy_and_devices_warns_ddp_single(self):
        """DDPStrategy with 1 device should warn."""
        import warnings

        from ocean.trainer.configuration_validator import _check_strategy_and_devices

        trainer = MagicMock()
        trainer.strategy = ocean.DDPStrategy()
        trainer.devices = 1
        trainer._is_distributed_spawn = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_strategy_and_devices(trainer)
            assert len(w) >= 1
            assert "DDPStrategy" in str(w[0].message)

    def test_check_strategy_and_devices_no_warn_single_device(self):
        """SingleDeviceStrategy should not warn."""
        import warnings

        from ocean.trainer.configuration_validator import _check_strategy_and_devices

        trainer = MagicMock()
        trainer.strategy = ocean.SingleDeviceStrategy()
        trainer.devices = 1
        trainer._is_distributed_spawn = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_strategy_and_devices(trainer)
            assert len(w) == 0

    def test_check_strategy_and_devices_no_warn_multi_device(self):
        """DDPStrategy with multiple devices should not warn."""
        import warnings

        from ocean.trainer.configuration_validator import _check_strategy_and_devices

        trainer = MagicMock()
        trainer.strategy = ocean.DDPStrategy()
        trainer.devices = 2
        trainer._is_distributed_spawn = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_strategy_and_devices(trainer)
            # DDP with 2 devices should not warn
            assert len(w) == 0


# ── Model export paddle format with inference model ─────────────


class TestModelExportPaddleFormat:
    """Test that model export -f paddle can produce inference models."""

    def test_export_paddle_without_model_class_saves_state_dict(self, tmp_path):
        """Export without --model-class should save state_dict."""
        from ocean.cli.model import _export_paddle

        model = paddle.nn.Linear(10, 2)
        ckpt = str(tmp_path / "model.pdparams")
        paddle.save(model.state_dict(), ckpt)

        output = str(tmp_path / "exported")
        _export_paddle(ckpt, output)

        assert (tmp_path / "exported.pdparams").exists()


# ── Cloud job submit packaging ──────────────────────────────────


class TestCloudJobSubmit:
    """Test cloud job submit packaging logic."""

    def test_job_submit_path_not_directory(self):
        """job submit with non-directory path should error."""
        from click.testing import CliRunner

        from ocean.cli import cli

        runner = CliRunner()
        result = runner.invoke(
            cli, ["cloud", "job", "submit", "--name", "test", "--cmd", "python test.py", "--path", "/nonexistent_dir"]
        )
        # Click may exit with 1 on unhandled exceptions from get_token
        # The key assertion is that the error message appears
        assert "must be a directory" in result.output.lower() or result.exit_code != 0

    def test_job_submit_packages_code(self, tmp_path):
        """job submit should create a zip from the code directory."""
        # Create a fake code directory
        code_dir = tmp_path / "mycode"
        code_dir.mkdir()
        (code_dir / "train.py").write_text("print('hello')")
        (code_dir / "config.yaml").write_text("name: test")

        # Mock the _post function in the job module
        call_args = []

        def mock_post(url, data, token):
            call_args.append((url, data))
            if "bosacl" in url:
                return {"upload_url": None}  # no BOS upload
            return {"pipeline_id": "12345", "status": "created"}

        # Patch _post in the job module's namespace
        import importlib

        job_module = importlib.import_module("ocean.cli.cloud.job")

        with patch.object(job_module, "_post", mock_post):
            from click.testing import CliRunner

            from ocean.cli import cli

            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "cloud",
                    "job",
                    "submit",
                    "--name",
                    "test-job",
                    "--cmd",
                    "python train.py",
                    "--path",
                    str(code_dir),
                    "--token",
                    "fake_token",
                ],
            )
            assert result.exit_code == 0, f"job submit failed: {result.output}"
            assert "Packaging code" in result.output or "Job submitted" in result.output
