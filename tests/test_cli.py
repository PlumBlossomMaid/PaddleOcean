"""Tests for CLI commands: ocean train, ocean model export/serve, ocean cloud."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import paddle
import pytest
from click.testing import CliRunner

from ocean.cli import cli
from ocean.cli.model import _import_class, _parse_input_spec

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def simple_model_file(tmp_path):
    """Create a test model module and checkpoint."""
    model_code = """
import ocean
import paddle


class SimpleModel(ocean.Model):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = paddle.nn.functional.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return paddle.optimizer.Adam(learning_rate=1e-3, parameters=self.parameters())
"""
    model_file = tmp_path / "test_model_module.py"
    model_file.write_text(model_code)

    # Add tmp_path to sys.path so we can import it
    sys.path.insert(0, str(tmp_path))

    # Create checkpoint
    from test_model_module import SimpleModel  # type: ignore[import-not-found]

    model = SimpleModel()
    ckpt_path = str(tmp_path / "test_checkpoint.pdparams")
    paddle.save(model.state_dict(), ckpt_path)

    yield {"model_file": str(model_file), "checkpoint": ckpt_path, "model_class": "test_model_module.SimpleModel"}

    sys.path.remove(str(tmp_path))


# ── CLI top-level ─────────────────────────────────────────────────


class TestCLITopLevel:
    def test_cli_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "train" in result.output
        assert "model" in result.output
        assert "cloud" in result.output

    def test_cli_no_command(self, runner):
        # Click groups exit with code 2 when no subcommand is given
        result = runner.invoke(cli, [])
        assert result.exit_code == 2  # Click's standard "missing command" exit code


# ── ocean train ───────────────────────────────────────────────────


class TestTrainCLI:
    def test_train_help(self, runner):
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--model" in result.output
        assert "--max-epochs" in result.output
        assert "--accelerator" in result.output
        assert "--devices" in result.output
        assert "--precision" in result.output
        assert "--checkpoint" in result.output

    def test_train_missing_model(self, runner):
        result = runner.invoke(cli, ["train", "--max-epochs", "1"])
        assert result.exit_code == 0
        assert "no model specified" in result.output.lower()

    def test_train_config_not_found(self, runner):
        result = runner.invoke(cli, ["train", "--config", "/nonexistent.yaml"])
        assert result.exit_code == 0
        assert "not found" in result.output.lower()

    def test_train_with_yaml_config(self, runner, simple_model_file, tmp_path):
        """Test training from a YAML config file."""
        config = {
            "model": {"_target_": "test_model_module.SimpleModel"},
            "trainer": {"max_epochs": 1, "accelerator": "cpu", "enable_progress_bar": False},
        }
        config_path = tmp_path / "config.yaml"
        import yaml

        config_path.write_text(yaml.dump(config))

        result = runner.invoke(cli, ["train", "--config", str(config_path)])
        assert result.exit_code == 0, f"train failed: {result.output}\nexception: {result.exception}"

    def test_train_with_model_class_flag(self, runner, simple_model_file):
        """Test training with --model flag."""
        result = runner.invoke(
            cli,
            [
                "train",
                "--model",
                simple_model_file["model_class"],
                "--max-epochs",
                "1",
                "--accelerator",
                "cpu",
                "--devices",
                "1",
            ],
        )
        assert result.exit_code == 0, f"train failed: {result.output}"


# ── ocean model export ────────────────────────────────────────────


class TestModelExportCLI:
    def test_export_help(self, runner):
        result = runner.invoke(cli, ["model", "export", "--help"])
        assert result.exit_code == 0
        assert "--checkpoint" in result.output
        assert "--format" in result.output
        assert "--output" in result.output
        assert "--model-class" in result.output
        assert "--input-shape" in result.output

    def test_export_paddle_format(self, runner, simple_model_file, tmp_path):
        """Test export to paddle format."""
        output_path = str(tmp_path / "exported_model")
        result = runner.invoke(
            cli,
            ["model", "export", "-c", simple_model_file["checkpoint"], "-f", "paddle", "-o", output_path],
        )
        assert result.exit_code == 0, f"export failed: {result.output}"
        assert os.path.exists(output_path + ".pdparams")

    def test_export_checkpoint_not_found(self, runner):
        result = runner.invoke(cli, ["model", "export", "-c", "/nonexistent.pdparams"])
        assert result.exit_code == 0
        assert "not found" in result.output.lower()

    def test_export_onnx_missing_model_class(self, runner, simple_model_file):
        result = runner.invoke(
            cli,
            ["model", "export", "-c", simple_model_file["checkpoint"], "-f", "onnx"],
        )
        assert result.exit_code == 0
        assert "--model-class is required" in result.output

    def test_export_onnx_missing_input_shape(self, runner, simple_model_file):
        result = runner.invoke(
            cli,
            [
                "model",
                "export",
                "-c",
                simple_model_file["checkpoint"],
                "-f",
                "onnx",
                "--model-class",
                simple_model_file["model_class"],
            ],
        )
        assert result.exit_code == 0
        assert "--input-shape is required" in result.output

    def test_export_onnx_invalid_model_class(self, runner, simple_model_file):
        result = runner.invoke(
            cli,
            [
                "model",
                "export",
                "-c",
                simple_model_file["checkpoint"],
                "-f",
                "onnx",
                "--model-class",
                "nonexistent_module.NonexistentClass",
                "--input-shape",
                "1,10",
            ],
        )
        assert result.exit_code == 0
        assert "cannot import" in result.output.lower() or "Error" in result.output


# ── ocean model serve ─────────────────────────────────────────────


class TestModelServeCLI:
    def test_serve_help(self, runner):
        result = runner.invoke(cli, ["model", "serve", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--port" in result.output
        assert "--host" in result.output
        assert "--model-class" in result.output

    def test_serve_pdparams_missing_model_class(self, runner, simple_model_file):
        """Serve .pdparams without --model-class should error."""
        result = runner.invoke(cli, ["model", "serve", "-m", simple_model_file["checkpoint"]])
        assert result.exit_code == 0
        assert "--model-class is required" in result.output

    def test_serve_pdparams_loads_model(self, runner, simple_model_file):
        """Test that serve loads a .pdparams model with --model-class."""
        # We can't actually run the server in a test, but we can verify
        # that the model loads correctly before the server starts.
        # Use a very short timeout via a mock.

        original_serve_forever = None
        try:
            # Mock HTTPServer.serve_forever to return immediately
            from http.server import HTTPServer

            original_serve_forever = HTTPServer.serve_forever
            HTTPServer.serve_forever = lambda self, *args, **kwargs: None

            result = runner.invoke(
                cli,
                [
                    "model",
                    "serve",
                    "-m",
                    simple_model_file["checkpoint"],
                    "--model-class",
                    simple_model_file["model_class"],
                    "--port",
                    "0",
                ],
            )
            assert result.exit_code == 0, f"serve failed: {result.output}"
            assert "Loaded state_dict" in result.output
        finally:
            if original_serve_forever:
                from http.server import HTTPServer

                HTTPServer.serve_forever = original_serve_forever


# ── ocean cloud ───────────────────────────────────────────────────


class TestCloudCLI:
    def test_cloud_help(self, runner):
        result = runner.invoke(cli, ["cloud", "--help"])
        assert result.exit_code == 0
        assert "login" in result.output
        assert "logout" in result.output
        assert "upload" in result.output
        assert "download" in result.output
        assert "list" in result.output
        assert "delete" in result.output
        assert "job" in result.output

    def test_cloud_login(self, runner):
        result = runner.invoke(cli, ["cloud", "login", "--token", "test_token_123"])
        assert result.exit_code == 0
        assert "Token saved" in result.output

    def test_cloud_logout(self, runner):
        # First login, then logout
        runner.invoke(cli, ["cloud", "login", "--token", "test_token_456"])
        result = runner.invoke(cli, ["cloud", "logout"])
        assert result.exit_code == 0
        assert "Token cleared" in result.output

    def test_cloud_list_help(self, runner):
        result = runner.invoke(cli, ["cloud", "list", "--help"])
        assert result.exit_code == 0
        assert "--repo-type" in result.output
        assert "--revision" in result.output

    def test_cloud_upload_help(self, runner):
        result = runner.invoke(cli, ["cloud", "upload", "--help"])
        assert result.exit_code == 0

    def test_cloud_download_help(self, runner):
        result = runner.invoke(cli, ["cloud", "download", "--help"])
        assert result.exit_code == 0

    def test_cloud_delete_help(self, runner):
        result = runner.invoke(cli, ["cloud", "delete", "--help"])
        assert result.exit_code == 0

    def test_cloud_job_help(self, runner):
        result = runner.invoke(cli, ["cloud", "job", "--help"])
        assert result.exit_code == 0
        assert "submit" in result.output
        assert "list" in result.output
        assert "stop" in result.output

    def test_cloud_job_submit_help(self, runner):
        result = runner.invoke(cli, ["cloud", "job", "submit", "--help"])
        assert result.exit_code == 0
        assert "--name" in result.output
        assert "--cmd" in result.output
        assert "--path" in result.output


# ── Helper functions ──────────────────────────────────────────────


class TestModelCLIHelpers:
    def test_parse_input_spec(self):
        assert _parse_input_spec("1,3,224,224") == [1, 3, 224, 224]
        assert _parse_input_spec("1,10") == [1, 10]
        assert _parse_input_spec("32") == [32]

    def test_import_class(self, simple_model_file):
        cls = _import_class(simple_model_file["model_class"])
        assert cls is not None
        assert hasattr(cls, "forward")


# ── ocean.cloud Python API ────────────────────────────────────────


class TestCloudPythonAPI:
    def test_import_cloud_functions(self):
        from ocean.cloud import (
            delete_file,
            download_file,
            get_token,
            get_token_optional,
            list_files,
            upload_file,
            upload_folder,
        )

        assert callable(upload_file)
        assert callable(upload_folder)
        assert callable(download_file)
        assert callable(delete_file)
        assert callable(list_files)
        assert callable(get_token)
        assert callable(get_token_optional)

    def test_get_token_optional_returns_none_when_no_token(self, monkeypatch):
        # Remove token file and env
        monkeypatch.delenv("AISTUDIO_ACCESS_TOKEN", raising=False)
        from ocean.cli.cloud.auth import TOKEN_FILE, get_token_optional

        if TOKEN_FILE.exists():
            monkeypatch.setattr("ocean.cli.cloud.auth.TOKEN_FILE", Path("/tmp/nonexistent_token_file"))

        assert get_token_optional() is None

    def test_get_token_raises_when_no_token(self, monkeypatch):
        monkeypatch.delenv("AISTUDIO_ACCESS_TOKEN", raising=False)
        from ocean.cli.cloud.auth import TOKEN_FILE, get_token

        if TOKEN_FILE.exists():
            monkeypatch.setattr("ocean.cli.cloud.auth.TOKEN_FILE", Path("/tmp/nonexistent_token_file"))

        import click

        with pytest.raises(click.ClickException):
            get_token()
