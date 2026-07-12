[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![PaddlePaddle](https://img.shields.io/badge/paddlepaddle-2.4%2B-brightgreen.svg)]()
[![tests](https://img.shields.io/badge/tests-44%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)]()

[![EN](https://img.shields.io/badge/lang-EN-red.svg)](README.md)
[![简体中文](https://img.shields.io/badge/lang-简体中文-blue.svg)](README.zh-CN.md)
[![繁體中文](https://img.shields.io/badge/lang-繁體中文-green.svg)](README.zh-TW.md)

# 🌊 PaddleOcean

> 对标 PyTorch Lightning 的 PaddlePaddle 高层框架 — Trainer · Model · Callbacks · Loggers · DDP · Gear · VisualDL。全部使用 PaddlePaddle 原生 API，零 PyTorch 依赖。

---

## 为什么用 PaddleOcean？

如果你用过 PyTorch Lightning，可以直接上手 PaddleOcean。每个核心模块都映射到 PaddlePaddle 的对应 API。

**Lightning 风格钩子：**

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

**Keras 风格快速原型：**

```python
net = paddle.nn.Sequential(...)
model = ocean.Model(__model__=net)
model.prepare(optimizer=opt, loss=loss_fn, metrics=[acc])
model.fit(train_loader, epochs=10)
```

---

## 功能覆盖

| 类别 | 覆盖 |
|------|------|
| **Model**（Keras + Lightning 双模式） | ✅ |
| **Trainer**（训练/验证/测试/预测） | ✅ |
| **DataModule**（数据生命周期） | ✅ |
| **Gear**（对标 Fabric，手动训练） | ✅ |
| **distributed**（70+ Paddle 分布式 API） | ✅ |
| **Callbacks**（18 种：断点、早停、计时器、SWA、Spike 检测等） | ✅ |
| **Loggers**（9 种：CSV、VisualDL、TensorBoard、Wandb、MLFlow、Comet、Ocelogger） | ✅ |
| **Strategies**（6 种：单卡、DDP、DeepSpeed、FSDP、模型并行） | ✅ |
| **Accelerators**（7 设备：CPU、CUDA、ROCm、XPU、IPU、自定义） | ✅ |
| **Precision**（4 种：全精度、AMP O1/O2、半精度、双精度） | ✅ |
| **CI**（3 OS × 4 Python 版本 × 代码风格检查） | ✅ |

与 `pytorch-lightning/` 功能对齐率约 98%（140+ 源文件覆盖）。

---

## 快速开始

```bash
pip install paddlepaddle
pip install -e .
pytest tests/ -v --timeout=120
```

```python
import ocean
print(ocean.__all__)  # 60+ 导出符号
```

---

## 项目结构

```
paddleOcean/
├── ocean/
│   ├── __init__.py, model.py, datamodule.py
│   ├── trainer/              完整训练引擎 + 连接器
│   ├── gear.py               手动训练 API
│   ├── distributed.py        70+ Paddle 分布式 API 封装
│   ├── callbacks/            18 个回调
│   ├── loggers/              9 个日志器
│   ├── strategies/           6 种策略
│   ├── accelerators/         7 种设备后端
│   ├── plugins/              精度 / IO / 环境
│   ├── profilers/            性能分析
│   ├── core/                 hooks, mixins, saving, optimizer
│   └── utils/                types, seed, rank_zero, exceptions, ...
├── tests/                    44 个测试 (pytest)
├── .github/workflows/        CI: ubuntu + windows + macOS
├── QWEN.md                   完整架构文档
└── pyproject.toml
```

---

## 许可证

Apache 2.0
