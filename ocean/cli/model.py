"""ocean model — export and serve trained models."""

from __future__ import annotations

import importlib
import os
from typing import Optional

import click


def _import_class(qualified_name: str):
    """Import 'module.path.ClassName' and return the class."""
    module_path, class_name = qualified_name.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _parse_input_spec(input_spec: str) -> list[int]:
    """Parse comma-separated shape string like '1,3,224,224'."""
    return [int(x) for x in input_spec.split(",")]


@click.group()
def model():
    """Export, serve, and manage trained models."""


@model.command()
@click.option("--checkpoint", "-c", required=True, help="Path to checkpoint file (.pdparams).")
@click.option("--format", "-f", type=click.Choice(["onnx", "paddle"]), default="paddle", help="Export format.")
@click.option("--output", "-o", default=None, help="Output path (file or directory).")
@click.option(
    "--model-class",
    type=str,
    default=None,
    help="Fully qualified model class, e.g. 'my_module.MyModel'. Required for ONNX export.",
)
@click.option(
    "--input-shape",
    type=str,
    default=None,
    help="Comma-separated input shape for export, e.g. '1,3,224,224'. Required for ONNX export.",
)
@click.option("--dynamic-shape", is_flag=True, help="Export with dynamic input shapes (ONNX only).")
def export(checkpoint, format, output, model_class, input_shape, dynamic_shape):
    """Export a trained model to ONNX or PaddlePaddle static graph.

    Examples:

        # Export to PaddlePaddle static graph (inference model)
        ocean model export -c checkpoint.pdparams -f paddle -o ./inference_model

        # Export to ONNX with a known input shape
        ocean model export -c checkpoint.pdparams -f onnx \\
            --model-class my_module.MyModel \\
            --input-shape 1,3,224,224 \\
            -o model.onnx
    """
    try:
        import paddle  # noqa: F401
    except ImportError:
        click.echo("Error: paddlepaddle is not installed.", err=True)
        return

    if not os.path.exists(checkpoint):
        click.echo(f"Error: checkpoint file '{checkpoint}' not found.", err=True)
        return

    if format == "paddle":
        _export_paddle(checkpoint, output)
    elif format == "onnx":
        _export_onnx(checkpoint, output, model_class, input_shape, dynamic_shape)


def _export_paddle(checkpoint: str, output: Optional[str]):
    """Export to PaddlePaddle inference model format."""
    import paddle

    if not output:
        output = checkpoint.rsplit(".", 1)[0]

    state_dict = paddle.load(checkpoint)
    click.echo(f"Loaded checkpoint with {len(state_dict)} parameters")

    # Save as static graph inference model if the checkpoint contains a state_dict
    paddle.save(state_dict, output + ".pdparams")
    click.echo(f"✅ Model saved to {output}.pdparams")


def _export_onnx(
    checkpoint: str, output: Optional[str], model_class: Optional[str], input_shape: Optional[str], dynamic_shape: bool
):
    """Export to ONNX format."""
    import paddle

    if not model_class:
        click.echo("Error: --model-class is required for ONNX export.", err=True)
        return
    if not input_shape:
        click.echo("Error: --input-shape is required for ONNX export (e.g. --input-shape 1,3,224,224).", err=True)
        return

    if not output:
        output = checkpoint.rsplit(".", 1)[0] + ".onnx"

    # Import and build model
    module_path, class_name = model_class.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        model_cls = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        click.echo(f"Error: cannot import model class '{model_class}': {e}", err=True)
        return

    model_instance = model_cls()

    # Load checkpoint
    state_dict = paddle.load(checkpoint)
    model_instance.set_state_dict(state_dict)
    model_instance.eval()

    # Build input spec
    shape = _parse_input_spec(input_shape)
    input_spec = [paddle.static.InputSpec(shape=shape, dtype="float32", name="input")]

    if dynamic_shape:
        # Replace fixed dimensions with -1 for dynamic export
        dyn_shape = [-1 if d == 1 else d for d in shape]
        if all(d == -1 for d in dyn_shape):
            dyn_shape = [-1] * len(shape)
        input_spec = [paddle.static.InputSpec(shape=dyn_shape, dtype="float32", name="input")]

    try:
        paddle.onnx.export(model_instance, output, input_spec=input_spec)
        click.echo(f"✅ Model exported to {output}.onnx")
    except Exception as e:
        click.echo(f"Error during ONNX export: {e}", err=True)


@model.command()
@click.option(
    "--model", "-m", required=True, help="Path to ONNX, PaddlePaddle inference model, or .pdparams checkpoint."
)
@click.option("--port", "-p", default=8501, type=int, help="Serving port.")
@click.option("--host", default="0.0.0.0", help="Serving host.")
@click.option(
    "--model-class", type=str, default=None, help="Model class for .pdparams serving, e.g. 'my_module.MyModel'."
)
def serve(model, port, host, model_class):
    """Serve a model via a simple HTTP API.

    The server loads the model and exposes a POST /predict endpoint
    that accepts JSON input and returns predictions.

    Supports three model formats:
    - ONNX (.onnx) — requires onnxruntime
    - PaddlePaddle inference model (.pdmodel + .pdiparams)
    - PaddlePaddle state_dict (.pdparams) — requires --model-class

    Example:

        ocean model serve --model model.onnx --port 8501

        ocean model serve --model checkpoint.pdparams --model-class my_model.MyModel
    """
    try:
        import numpy as np
    except ImportError:
        click.echo("Error: numpy is required for serving.", err=True)
        return

    try:
        import json
        from http.server import BaseHTTPRequestHandler, HTTPServer

        # Try to load the model
        model_loaded = False
        serve_mode = None  # "onnx", "inference", "state_dict"

        if model.endswith(".onnx"):
            try:
                import onnxruntime as ort

                session = ort.InferenceSession(model)
                model_loaded = True
                serve_mode = "onnx"
                input_name = session.get_inputs()[0].name
                click.echo(f"Loaded ONNX model, input: {input_name}")
            except ImportError:
                click.echo(
                    "Error: onnxruntime is required to serve ONNX models. Install with: pip install onnxruntime",
                    err=True,
                )
                return
        elif model.endswith(".pdparams"):
            if not model_class:
                click.echo("Error: --model-class is required to serve a .pdparams checkpoint.", err=True)
                return
            try:
                import paddle

                cls = _import_class(model_class)
                model_instance = cls()
                state_dict = paddle.load(model)
                model_instance.set_state_dict(state_dict)
                model_instance.eval()
                model_loaded = True
                serve_mode = "state_dict"
                click.echo(f"Loaded state_dict into {type(model_instance).__name__}")
            except Exception as e:
                click.echo(f"Error loading model: {e}", err=True)
                return
        else:
            try:
                import paddle

                # Load PaddlePaddle inference model
                config = paddle.inference.Config(model + ".pdmodel", model + ".pdiparams")
                predictor = paddle.inference.create_predictor(config)
                model_loaded = True
                serve_mode = "inference"
                click.echo("Loaded PaddlePaddle inference model")
            except Exception as e:
                click.echo(f"Error loading PaddlePaddle model: {e}", err=True)
                return

        if not model_loaded:
            click.echo(f"Error: could not load model '{model}'", err=True)
            return

        class PredictionHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path != "/predict":
                    self.send_error(404, "Not found")
                    return
                try:
                    content_length = int(self.headers["Content-Length"])
                    body = self.rfile.read(content_length)
                    data = json.loads(body)

                    # Convert input to numpy array
                    if "input" in data:
                        arr = np.array(data["input"], dtype="float32")
                    else:
                        self.send_error(400, "Missing 'input' field")
                        return

                    if serve_mode == "onnx":
                        result = session.run(None, {input_name: arr})
                    elif serve_mode == "state_dict":
                        import paddle

                        tensor = paddle.to_tensor(arr)
                        with paddle.no_grad():
                            output_tensor = model_instance(tensor)
                        result = [output_tensor.numpy()]
                    else:
                        input_names = predictor.get_input_names()
                        input_handle = predictor.get_input_handle(input_names[0])
                        input_handle.copy_from_cpu(arr)
                        predictor.run()
                        output_names = predictor.get_output_names()
                        output_handle = predictor.get_output_handle(output_names[0])
                        result = [output_handle.copy_to_cpu()]

                    # Convert numpy arrays to lists for JSON
                    output = [r.tolist() if hasattr(r, "tolist") else r for r in result]
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"prediction": output}).encode())

                except Exception as e:
                    self.send_error(500, f"Prediction error: {e}")

            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok"}).encode())
                elif self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(
                        b"<html><body><h1>Ocean Model Server</h1>"
                        b'<p>POST to /predict with JSON: {"input": [[...]]}</p>'
                        b"<p>GET /health for health check</p>"
                        b"</body></html>"
                    )
                else:
                    self.send_error(404, "Not found")

            def log_message(self, format, *args):
                pass  # Suppress default logging

        server = HTTPServer((host, port), PredictionHandler)
        click.echo(f"🚀 Serving {model} on http://{host}:{port}")
        click.echo("   POST /predict  - Get predictions")
        click.echo("   GET  /health   - Health check")
        click.echo("   Press Ctrl+C to stop")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            click.echo("\n👋 Server stopped")
            server.server_close()

    except ImportError as e:
        click.echo(f"Error: missing dependency: {e}", err=True)
