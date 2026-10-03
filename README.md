[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![PaddlePaddle](https://img.shields.io/badge/paddlepaddle-2.4%2B-brightgreen.svg)]()
[![tests](https://img.shields.io/badge/tests-538%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)]()
[![PyPI](https://img.shields.io/badge/PyPI-0.1.0-blue.svg)]()

# 🌊 PaddleOcean

**A high-level PaddlePaddle framework inspired by PyTorch Lightning**

Trainer · Model · Callbacks · Loggers · DDP · Gear · VisualDL · CLI

---

## Why PaddleOcean?

PaddleOcean maps every core component of PyTorch Lightning to PaddlePaddle's native API, with zero PyTorch dependencies. If you know Lightning, you already know PaddleOcean.

**Lightning-style hooks:**

```python
import paddle
import ocean

class MyModel(ocean.Model):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(28, 10)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = paddle.nn.functional.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return paddle.optimizer.Adam(learning_rate=1e-3, parameters=self.parameters())

model = MyModel()
trainer = ocean.Trainer(max_epochs=10, accelerator="gpu")
trainer.fit(model, train_loader, val_loader)
```

**Keras-style quick prototyping:**

```python
net = paddle.nn.Sequential(paddle.nn.Flatten(), paddle.nn.Linear(28, 10))
model = ocean.Model(__model__=net)
model.prepare(optimizer=opt, loss=loss_fn, metrics=[acc])
model.fit(train_loader, epochs=10)
```

**CLI training:**

```bash
# Train from YAML config
ocean train --config config.yaml --accelerator gpu

# Export model to ONNX
ocean model export -c checkpoint.pdparams -f onnx --model-class my_model.MyModel --input-shape 1,3,224,224

# Serve model via HTTP
ocean model serve --model model.onnx --port 8501
```

**Cloud integration (AI Studio):**

```bash
ocean cloud login --token <token>
ocean cloud upload user/repo ./data.zip
ocean cloud download user/repo
ocean cloud list user/repo
```

---

## Features

| Category | Coverage |
|----------|----------|
| **Model** (Keras + Lightning dual-mode) | ✅ |
| **Trainer** (fit / validate / test / predict) | ✅ |
| **DataModule** (data lifecycle) | ✅ |
| **Gear** (Fabric equivalent, manual training) | ✅ |
| **distributed** (70+ Paddle distributed APIs) | ✅ |
| **Callbacks** (20 kinds: Checkpoint, EarlyStopping, Timer, SWA, LRFinder, ...) | ✅ |
| **Loggers** (9 kinds: CSV, VisualDL, TensorBoard, Wandb, MLFlow, Comet, ...) | ✅ |
| **Strategies** (6: SingleDevice, DDP, DeepSpeed, FSDP, ModelParallel) | ✅ |
| **Accelerators** (7 devices: CPU, CUDA, ROCm, XPU, IPU, CustomDevice) | ✅ |
| **Precision** (4: 32, AMP O1/O2, Half, Double) | ✅ |
| **Profilers** (3: Simple, Advanced, PassThrough) | ✅ |
| **CLI** (train, model export/serve, cloud upload/download/list/delete/job) | ✅ |
| **CI** (3 OS × 4 Python versions × lint) | ✅ |

---

## Installation

```bash
pip install paddlepaddle-gpu  # or paddlepaddle for CPU
pip install paddleocean
```

### From source

```bash
git clone https://github.com/PlumBlossomMaid/PaddleOcean.git
cd PaddleOcean
pip install -e .
```

---

## Quick Start

```python
import ocean
print(ocean.__all__)  # 100+ exported symbols
```

---

## Project Structure

```
PaddleOcean/
├── ocean/
│   ├── __init__.py          # 100+ public API exports
│   ├── model.py             # Model (Keras + Lightning dual-mode)
│   ├── trainer/             # Full engine + connectors
│   ├── gear.py              # Manual training API (Fabric equivalent)
│   ├── distributed.py       # 70+ Paddle distributed API wrapper
│   ├── callbacks/           # 20 callbacks
│   ├── loggers/             # 9 loggers
│   ├── strategies/          # 6 strategies
│   ├── accelerators/        # 7 device backends
│   ├── plugins/             # precision / IO / environments / layer_sync
│   ├── profilers/           # profiling tools
│   ├── core/                # hooks, mixins, saving, optimizer
│   ├── cli/                 # ocean train / model / cloud
│   └── utils/               # types, seed, rank_zero, compile, registry, ...
├── tests/                   # 538 tests (pytest)
├── .github/workflows/       # CI: ubuntu + windows + macOS
├── QWEN.md                  # Full architecture documentation
└── pyproject.toml
```

---

## License

Apache 2.0
