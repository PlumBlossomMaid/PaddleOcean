# paddleOcean — PaddlePaddle 高层框架

对标 PyTorch Lightning，100% 复刻其功能体系，全部使用 PaddlePaddle 原生 API。

## 架构全景

```
ocean/                          lightning/pytorch/
├── __init__.py                 导出 60+ 符号
│
├── Model                       核心：双模式（Keras + Lightning）
│   ├── Keras 模式: __model__ + compile() + fit()
│   └── Lightning 模式: hooks (training_step, configure_optimizers ...)
│
├── Trainer                     训练引擎
│   ├── connectors/             数据/日志/回调/检查点/信号/加速器连接器
│   ├── loops/                  训练/验证/测试/预测循环
│   ├── call.py                 钩子分发
│   └── states.py               状态管理
│
├── DataModule                  数据生命周期
├── Gear                        轻量手动训练（对标 Fabric）
├── distributed.py              70+ Paddle 分布式 API 封装
│
├── callbacks/                  18 个回调
│   ├── Callback                基类
│   ├── ModelCheckpoint         自动断点
│   ├── EarlyStopping           早停
│   ├── └── ...                 计时器/LR 监控/SWA/...
│
├── loggers/                    9 个日志器
│   ├── CSVLogger               文件日志
│   ├── VisualDLLogger          Paddle 原生可视化
│   ├── TensorBoardLogger       TensorBoard 格式
│   ├── Wandb/MLFlow/Comet      第三方
│   └── OceanLogger             统一包装
│
├── strategies/                 6 种策略
│   ├── SingleDevice            单卡
│   ├── DDP                     数据并行
│   ├── DeepSpeed/FSDP          大模型分片
│   └── ModelParallel           模型并行
│
├── accelerators/               7 种设备
│   ├── CPU/CUDA                GPU
│   ├── ROCm                    AMD
│   ├── XPU                     昆仑
│   ├── IPU                     Graphcore
│   └── CustomDevice            昇腾等
│
├── plugins/                    精度/IO/环境插件
├── profilers/                  性能分析
├── tuner/                      超参调优
└── utils/                      工具函数
```

## 设计原则

### 1. 无 `**kwargs`
所有参数必须显式声明。不透传未知参数。保证 API 透明、可文档化、类型安全。

### 2. 双模式 Model
```python
# Keras 模式（简单快速）
net = paddle.nn.Sequential(...)
model = ocean.Model(__model__=net)
model.compile(optimizer=opt, loss=loss_fn, metrics=[acc])
model.fit(train_loader, epochs=10)

# Lightning 模式（完整控制）
class MyModel(ocean.Model):
    def training_step(self, batch, batch_idx): ...
    def configure_optimizers(self): ...
model = MyModel()
trainer = ocean.Trainer(max_epochs=10)
trainer.fit(model, train_loader)
```

### 3. Paddle 原生命名
- `utils/` 而非 `utilities/`
- `set_state_dict()` 是标准方法，`load_state_dict()` 是别名
- 设备体系用 Paddle 的 `CPUPlace`/`CUDAPlace`/`XPUPlace`/`IPUPlace`/`CustomPlace`
- 精度用 `paddle.float16`/`bfloat16`/`float64`
- AMP 用 `paddle.amp.GradScaler` / `auto_cast`

### 4. 无 Lightning 过时成分
跳过 `_graveyard/`、`neptune`、`pruning`、TPU/XLA/MPS 等平台独占或已废弃模块。

### 5. `import ocean` 风格
统一 `ocean.xxx` 命名空间，不搞 `from ocean import xxx` 碎片化。

### 6. 所有 `Lightning*` 名称 → `Ocean*`
`LightningEnum` → `OceanEnum`，`LitLogger` → `Ocelogger`，以此类推。

## 模块建设指导

### Model
- **核心职责**：作为用户的主要入口，既支持快速原型（Keras 模式）也支持完整控制（Lightning 模式）
- **关键设计**：`__model__` 为 None 时走 Lightning 模式，非 None 时走 Keras 模式
- **关键方法**：`training_step`/`validation_step`/`test_step`/`predict_step`/`configure_optimizers` 是 Lightning 模式核心钩子
- **扩展点**：`on_fit_start`/`on_train_epoch_end`/`on_before_backward` 等 30+ 生命周期钩子
- **数据流**：`batch → training_step → loss → backward → optimizer.step`

### Trainer
- **核心职责**：编排训练/验证/测试/预测循环，管理策略/加速器/精度/日志/回调
- **架构**：Trainer 本体很瘦，通过 6 个 Connector 代理到各子系统
- **关键路径**：`fit → _fit_impl → strategy.connect → data_connector.attach → fit_loop.run`
- **扩展点**：通过 callbacks 插入任意自定义行为
- **状态机**：`TrainerStatus.INITIALIZING → RUNNING → FINISHED/INTERRUPTED`

### Callbacks
- **设计模式**：观察者模式，所有钩子默认空实现
- **钩子签名**：`(self, trainer, model, ...)` — trainer 和 model 总是前两个参数
- **关键回调**：`ModelCheckpoint`（自动保存）、`EarlyStopping`（早停）、`Timer`（限时）
- **实现指导**：新回调继承 `Callback`，只重写需要的钩子，保持幂等

### Loggers
- **接口**：所有 Logger 实现 `log_metrics(metrics, step)` 和 `log_hyperparams(params)`
- **日志路径**：`{root_dir}/{name}/version_{N}/metrics.csv`
- **Ocean 特有**：`VisualDLLogger` 基于 Paddle 原生 VisualDL，`OceanLogger` 统一包装多 Logger

### Strategies
- **职责**：管理模型放置（device）、分布式包装（DDP）、精度上下文
- **DDP 流程**：`setup → accelerator.setup → precision.convert_module → model_to_device → DataParallel → setup_optimizers`
- **新增策略**：继承 `Strategy`，实现 `root_device`/`is_global_zero`/`setup`/`teardown`

### Accelerators
- **Paddle 设备体系**：`CPUPlace` / `CUDAPlace` / `XPUPlace` / `IPUPlace` / `CustomPlace(device_type)`
- **检测 API**：`paddle.is_compiled_with_cuda()` / `is_compiled_with_rocm()` / `is_compiled_with_xpu()`

### Gear
- **对标**：Lightning Fabric
- **适用场景**：需要手动控制训练循环但想要自动设备/精度/检查点管理的用户
- **关键差异**：Gear 不接管训练循环，只提供 `setup`/`backward`/`save`/`load` 等基础设施

## CI 配置

- **触发器**：push/PR 到 master/release/*，仅改动 ocean/tests/pyproject.toml/.github 时触发
- **任务**：lint → core-tests (3 OS × 4 Python) → distributed-tests → import-sanity
- **CPU only**：所有 runner 安装 `paddlepaddle`（CPU 版），GPU 测试在 Linux 环境手动执行
- **代码风格**：ruff v0.15.0（与 Paddle 主框架一致）
- **跳过条件**：draft PR 不触发

## 开发指南

```bash
# 安装
pip install paddlepaddle
pip install -e .[dev]

# 测试
pytest tests/ -v --timeout=120

# 代码风格
ruff check ocean/ tests/
ruff format ocean/ tests/ --check
pre-commit run --all-files

# 本地 CI 模拟
python -m pytest tests/ -v --timeout=120 -x
```
