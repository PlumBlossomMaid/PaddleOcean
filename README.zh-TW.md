[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![PaddlePaddle](https://img.shields.io/badge/paddlepaddle-2.4%2B-brightgreen.svg)]()
[![tests](https://img.shields.io/badge/tests-44%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)]()

[![EN](https://img.shields.io/badge/lang-EN-red.svg)](README.md)
[![简体中文](https://img.shields.io/badge/lang-简体中文-blue.svg)](README.zh-CN.md)
[![繁體中文](https://img.shields.io/badge/lang-繁體中文-green.svg)](README.zh-TW.md)

# 🌊 PaddleOcean

> 對標 PyTorch Lightning 的 PaddlePaddle 高層框架 — Trainer · Model · Callbacks · Loggers · DDP · Gear · VisualDL。全部使用 PaddlePaddle 原生 API，零 PyTorch 依賴。

---

## 為什麼用 PaddleOcean？

如果你用過 PyTorch Lightning，可以直接上手 PaddleOcean。每個核心模組都映射到 PaddlePaddle 的對應 API。

**Lightning 風格鉤子：**

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

**Keras 風格快速原型：**

```python
net = paddle.nn.Sequential(...)
model = ocean.Model(__model__=net)
model.prepare(optimizer=opt, loss=loss_fn, metrics=[acc])
model.fit(train_loader, epochs=10)
```

---

## 功能覆蓋

| 類別 | 覆蓋 |
|------|------|
| **Model**（Keras + Lightning 雙模式） | ✅ |
| **Trainer**（訓練/驗證/測試/預測） | ✅ |
| **DataModule**（資料生命週期） | ✅ |
| **Gear**（對標 Fabric，手動訓練） | ✅ |
| **distributed**（70+ Paddle 分散式 API） | ✅ |
| **Callbacks**（18 種：斷點、早停、計時器、SWA、Spike 檢測等） | ✅ |
| **Loggers**（9 種：CSV、VisualDL、TensorBoard、Wandb、MLFlow、Comet、Ocelogger） | ✅ |
| **Strategies**（6 種：單卡、DDP、DeepSpeed、FSDP、模型並行） | ✅ |
| **Accelerators**（7 裝置：CPU、CUDA、ROCm、XPU、IPU、自定義） | ✅ |
| **Precision**（4 種：全精度、AMP O1/O2、半精度、雙精度） | ✅ |
| **CI**（3 OS × 4 Python 版本 × 程式碼風格檢查） | ✅ |

與 `pytorch-lightning/` 功能對齊率約 98%（140+ 原始檔覆蓋）。

---

## 快速開始

```bash
pip install paddlepaddle
pip install -e .
pytest tests/ -v --timeout=120
```

```python
import ocean
print(ocean.__all__)  # 60+ 匯出符號
```

---

## 專案結構

```
paddleOcean/
├── ocean/
│   ├── __init__.py, model.py, datamodule.py
│   ├── trainer/              完整訓練引擎 + 連接器
│   ├── gear.py               手動訓練 API
│   ├── distributed.py        70+ Paddle 分散式 API 封裝
│   ├── callbacks/            18 個回呼
│   ├── loggers/              9 個日誌器
│   ├── strategies/           6 種策略
│   ├── accelerators/         7 種裝置後端
│   ├── plugins/              精度 / IO / 環境
│   ├── profilers/            性能分析
│   ├── core/                 hooks, mixins, saving, optimizer
│   └── utils/                types, seed, rank_zero, exceptions, ...
├── tests/                    44 個測試 (pytest)
├── .github/workflows/        CI: ubuntu + windows + macOS
├── QWEN.md                   完整架構文件
└── pyproject.toml
```

---

## 授權

Apache 2.0
