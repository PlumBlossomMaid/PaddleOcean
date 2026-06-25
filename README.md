[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![PaddlePaddle](https://img.shields.io/badge/paddlepaddle-2.4%2B-brightgreen.svg)]()
[![tests](https://img.shields.io/badge/tests-44%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)]()

[![EN](https://img.shields.io/badge/lang-EN-red.svg)](README.md)
[![简体中文](https://img.shields.io/badge/lang-简体中文-blue.svg)](README.zh-CN.md)
[![繁體中文](https://img.shields.io/badge/lang-繁體中文-green.svg)](README.zh-TW.md)

# 🌊 PaddleOcean

**A high-level PaddlePaddle framework inspired by PyTorch Lightning**

Trainer · Model · Callbacks · Loggers · DDP · Gear · VisualDL

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
trainer = ocean.Trainer(max_epochs=10)
trainer.fit(model, train_loader, val_loader)
```

**Keras-style quick prototyping:**

```python
net = paddle.nn.Sequential(...)
model = ocean.Model(__model__=net)
model.prepare(optimizer=opt, loss=loss_fn, metrics=[acc])
model.fit(train_loader, epochs=10)
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
| **Callbacks** (18 kinds: Checkpoint, EarlyStopping, Timer, SWA, SpikeDetection, ...) | ✅ |
| **Loggers** (9 kinds: CSV, VisualDL, TensorBoard, Wandb, MLFlow, Comet, Ocelogger) | ✅ |
| **Strategies** (6: SingleDevice, DDP, DeepSpeed, FSDP, ModelParallel) | ✅ |
| **Accelerators** (7 devices: CPU, CUDA, ROCm, XPU, IPU, CustomDevice) | ✅ |
| **Precision** (4: 32, AMP O1/O2, Half, Double) | ✅ |
| **CI** (3 OS × 4 Python versions × lint) | ✅ |

~98% feature parity with `pytorch-lightning/src/lightning/pytorch/` (140+ source files).

---

## Quick Start

```bash
pip install paddlepaddle
pip install -e .
pytest tests/ -v --timeout=120
```

```python
import ocean
print(ocean.__all__)  # 60+ exported symbols
```

---

## Project Structure

```
paddleOcean/
├── ocean/
│   ├── __init__.py, model.py, datamodule.py
│   ├── trainer/              # Full engine + connectors
│   ├── gear.py               # Manual training API
│   ├── distributed.py        # 70+ Paddle distributed API wrapper
│   ├── callbacks/            # 18 callbacks
│   ├── loggers/              # 9 loggers
│   ├── strategies/           # 6 strategies
│   ├── accelerators/         # 7 device backends
│   ├── plugins/              # precision / IO / environments
│   ├── profilers/            # profiling tools
│   ├── core/                 # hooks, mixins, saving, optimizer
│   └── utils/                # types, seed, rank_zero, exceptions, ...
├── tests/                    # 44 tests (pytest)
├── .github/workflows/        # CI: ubuntu + windows + macOS
├── QWEN.md                   # Full architecture documentation
└── pyproject.toml
```

---

## License

Apache 2.0
