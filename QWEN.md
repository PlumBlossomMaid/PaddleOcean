# paddleOcean — PaddlePaddle 高层框架

对标 PyTorch Lightning，100% 复刻其功能体系，全部使用 PaddlePaddle 原生 API。

## 架构全景

```
ocean/                          lightning/pytorch/
├── __init__.py                 导出 60+ 符号 + __getattr__ 代理所有 paddle API
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
├── gear_wrappers.py            _FabricModule / _FabricOptimizer
├── cli.py                      CLI 命令行启动
├── distributed.py              70+ Paddle 分布式 API 封装
│
├── _compat/                    多版本兼容层
│   ├── version.py              版本检测（2.4~3.3）
│   └── tensor.py               Tensor 操作 fallback（repeat_interleave, sort 等）
│
├── callbacks/                  18 个回调
│   ├── Callback                基类（30+ 钩子）
│   ├── ModelCheckpoint         自动断点（top-k / last / every N）
│   ├── EarlyStopping           早停
│   ├── Timer                   限时停止
│   ├── └── ...                 学习率监控 / SWA / Spike 检测 / ...
│
├── loggers/                    9 个日志器
│   ├── CSVLogger               文件日志
│   ├── VisualDLLogger          Paddle 原生可视化
│   ├── TensorBoardLogger       TensorBoard 格式（VisualDL 后端）
│   ├── Wandb/MLFlow/Comet      第三方
│   └── OceanLogger/Ocelogger   统一包装
│
├── strategies/                 6 种策略
│   ├── SingleDevice            单卡
│   ├── DDP                     数据并行（paddle.distributed）
│   ├── DeepSpeed/FSDP          大模型分片（group_sharded_parallel）
│   └── ModelParallel           模型并行（ProcessMesh）
│
├── accelerators/               7 种设备
│   ├── CPU/CUDA                NVIDIA GPU
│   ├── ROCm                    AMD GPU
│   ├── XPU                     百度昆仑
│   ├── IPU                     Graphcore
│   └── CustomDevice            昇腾 / 寒武纪等
│
├── plugins/                    精度/IO/环境插件
│   ├── precision/              全精度 / AMP O1O2 / Half / Double
│   ├── io/                     CheckpointIO / AsyncIO
│   └── environments/           集群环境
│
├── profilers/                  性能分析（Simple / Advanced via paddle.profiler）
├── tuner/                      LR range test / batch size scaler
├── core/                       hooks / mixins / saving / optimizer
├── cli/                        CLI 子系统
│   └── cloud/                  AI Studio 云 SDK（上传/下载/认证/任务）
└── utils/                      工具函数
    ├── seed.py                 seed_everything
    ├── rank_zero.py            rank_zero_only 装饰器
    ├── enums.py                OceanEnum
    ├── compile.py              paddle.jit.to_static（CINN）
    ├── model_summary/          模型摘要数据结构
    ├── testing/                条件测试跳过（@RunIf）
    └── migration/              断点版本迁移
```

## 设计原则

### 1. 零 `import paddle`（使用 `import ocean` 即可）
`ocean` 通过 `_PaddleProxy` 动态代理所有 `paddle.*` API。用户只需要：
```python
import ocean
x = ocean.randn([3, 4])              # paddle.randn
layer = ocean.nn.Linear(10, 2)       # paddle.nn.Linear
opt = ocean.optimizer.Adam(...)       # paddle.optimizer.Adam
loss = ocean.nn.functional.cross_entropy(...)
trainer = ocean.Trainer(max_epochs=10)
```

### 2. 多版本兼容（Paddle 2.4~3.3）
`ocean._compat` 自动检测 Paddle 版本，对旧版本缺失的 API 提供纯 Python fallback：
```python
ocean.repeat_interleave(x, 3)   # 2.5+ 用原生，旧版自动 fallback
ocean.sort(x, axis=-1)          # 返回 (values, indices)
ocean.unique(x)                 # 兼容任意版本
```

### 3. 双模式 Model
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

### 4. 无 `**kwargs`
所有参数必须显式声明。不透传未知参数。

### 5. Paddle 原生命名
`utils/` 而非 `utilities/`、`OceanEnum` 而非 `LightningEnum`、`set_state_dict` + `load_state_dict` 别名

### 6. 无 Lightning 过时成分
跳过 Neptune、Pruning、TPU/XLA/MPS 等已废弃或平台独占模块。

## 模块建设指导

### _compat（多版本兼容层）
- **核心职责**：检测 Paddle 版本，对低版本缺失的 API 提供纯 Python fallback
- **关键文件**：`version.py`（`Version` / `version_gte` / `api_available`）、`tensor.py`（fallback 实现）
- **添加新 fallback**：检查 `api_available("paddle.xxx")` → 不存在则实现纯 Python 版本
- **测试策略**：`from ocean._compat.tensor import xxx` 测试各边界条件

### Model
- **数据流**：`batch → training_step → loss → backward → optimizer.step`
- **30+ 生命周期钩子**：`on_fit_start` / `on_train_epoch_end` / `on_before_backward` ...
- **checkpoint**：`save_checkpoint(path)` / `load_checkpoint(path)` / `load_state_dict(sd, strict)`

### Trainer
- **架构**：Trainer 本体很瘦，通过 6 个 Connector 代理到各子系统
- **关键路径**：`fit → _fit_impl → strategy.connect → data_connector.attach → fit_loop.run`
- **状态机**：`TrainerStatus.INITIALIZING → RUNNING → FINISHED/INTERRUPTED`
- **加速器自动选择**：GPU 可用自动用 GPU，否则 CPU

### Callbacks
- **钩子签名**：`(self, trainer, model, *args)` — trainer 和 model 总是前两个参数
- **幂等性**：多次调用 `setup`/`teardown` 不应产生副作用

### Metrics
- **来源**：重导出自 `paddlemetrics`（对标 `torchmetrics`，从 TorchMetrics 移植）
- **导入方式**：领域指标直接 `from paddlemetrics import Accuracy, ...`（对齐 Lightning 不重新导出的模式）
- **核心类型**：`Metric`、`CompositionalMetric`、`MetricCollection` 在 `ocean.metrics` 中可用
- **`self.log()` 集成**：传递 `Metric` 对象给 `self.log()` 时，框架自动：
  - `on_step`：使用 `metric._forward_cache` 记录 batch 值
  - `on_epoch`：调用 `metric.compute()` 获取 epoch 值
  - `reduce_fx` 被忽略（Metric 自行处理归约）——对齐 Lightning
  - 分布式同步由 `paddlemetrics.Metric.sync()` 内部处理
- **多卡**：每个 Metric 内置 `dist_sync_on_step` / `sync_on_compute` 控制

### Loggers
- **路径结构**：`{root_dir}/{name}/version_{N}/metrics.csv`
- **Ocean 特有**：`VisualDLLogger`（VisualDL）、`OceanLogger`（统一包装）
- **多卡保护（对齐 Lightning）**：所有 7 个 logger 均使用 `@rank_zero_only` 和 `@rank_zero_experiment` 保护，确保非 rank 0 进程不执行写操作。详见 PR #5。

### Strategies
- **DDP 流程**：`setup → accelerator.setup → precision.convert_module → model_to_device → DataParallel → setup_optimizers`
- **新增策略**：继承 `Strategy`，实现 `root_device`/`is_global_zero`/`setup`/`teardown`

### Accelerators
- **Paddle 设备体系**：`CPUPlace` / `CUDAPlace` / `XPUPlace` / `IPUPlace` / `CustomPlace(device_type)`
- **检测**：`paddle.is_compiled_with_cuda()` / `is_compiled_with_rocm()` / `is_compiled_with_xpu()`
- **XPU vs CustomDevice 差异**：
  - XPU 硬编码在 Paddle 主框架中（不是 CustomDevice 插件），通过 `is_compiled_with_xpu()` 检测
  - 昇腾/寒武纪/天数等国产卡走 `CustomDevice` + `paddle.CustomPlace(device_type)`
  - `paddle.device.get_all_custom_device_type()` **不会**返回 `"xpu"`
- **XPU P800 适配文档**：见 `.qwen/skills/xpu-p800-setup.md` 及下方「XPU P800 环境」章节

### Cloud SDK（AI Studio）
`ocean/cli/cloud/` — 百度 AI Studio 云 SDK，上传/下载/认证/任务管理。

**入口：** CLI `ocean cloud <command>` / Python `from ocean.cloud import upload_file`

**详细文档：** 见 skill `/skill aistudio-cloud-upload`（含架构要点、历史修复经验）

### Gear
- **对标**：Lightning Fabric
- **适用场景**：需要手动控制循环但想要自动设备/精度/检查点管理

## CI 配置

- **触发器**：push/PR 到 master/release/\*，仅改动 ocean/tests/pyproject.toml/.github 时触发
- **任务**：lint (ruff 0.15.0) → core-tests (ubuntu + windows × Python 3.9~3.12) → import-sanity
- **CPU only**：CI runner 安装 `paddlepaddle`（CPU 版）
- **GPU 测试**：在本地 Linux 环境手动执行
- **跳过条件**：draft PR 不触发

## Training Behavior（训练行为详解）

详见 `.qwen/training-behavior.md`，涵盖：

- Sanity Check 行为（`_sanity_check()` 流程步骤）
- 训练循环（FitLoop）中的日志 flush 时机
- 验证流程（`_run_validation`）中的 metrics flush 与重置
- LoggerConnector 的 `reset_validation_metrics()` 原理
- VisualDLLogger 的 `version="latest"` 模式
- SOT KeyError 'self' 补丁机制
- 手动优化（`automatic_optimization=False`）的处理
- 常见 VDL 日志问题排查

## 本地验证

```bash
# 安装（已安装 paddle 后）
pip install -e . --no-build-isolation

# 全量测试
pytest tests/ -v --timeout=120

# 体验 demo
python ocean_demo.py --epochs 3

# 代码风格（与 Paddle 主框架一致，ruff v0.15.0）
ruff check .
ruff format .
```

## GPU 环境

```bash
pip uninstall paddlepaddle -y
pip install paddlepaddle-gpu
pip install -e . --no-build-isolation
pytest tests/ -v --timeout=120   # 57 tests, GPU 自动使用
python ocean_demo.py --epochs 3  # 三种模式演示
```

## XPU P800 环境（昆仑芯）

### 硬件概述

昆仑芯 P800 是百度自研 AI 加速卡：
- **显存**：32 GB HBM2e
- **Paddle 集成方式**：硬编码在主框架中（非 CustomDevice 插件）
- **检测 API**：`paddle.is_compiled_with_xpu()`

### 安装

```bash
# nightly 版本
python -m pip install --pre paddlepaddle-xpu \
  -i https://www.paddlepaddle.org.cn/packages/nightly/xpu-p800/

# 稳定版本
python -m pip install paddlepaddle-xpu \
  -i https://www.paddlepaddle.org.cn/packages/stable/xpu-p800/
```

验证：
```bash
python -c "import paddle; print(paddle.is_compiled_with_xpu())"  # → True
```

### 必需环境变量

```bash
export XPU_FORCE_USERMODE_LAUNCH=1
export XBLAS_FC_HBM_VERSION=40
export XPU_CDNN_CLUSTER_PARALLEL=1
export XPU_CDNN_CLUSTER_PARALLEL_STREAM_NUMBER=2
export XPU_PADDLE_L3_SIZE0=1024
export XPU_PADDLE_L3_SIZE1=1024
export XPUAPI_DEFAULT_SIZE0=1502653248
export XPUAPI_DEFAULT_SIZE1=380265324
export FLAGS_set_to_1d=False
export FLAGS_use_stride_kernel="0"
```

### Ocean 使用方式

```python
import ocean

trainer = ocean.Trainer(accelerator="xpu", max_epochs=10)
trainer.fit(model, train_loader)

# 手动
acc = ocean.XPUAccelerator()
print(acc.is_available())  # → True
```

### 与 GPU 差异

- GPU: `paddle.set_device("gpu")` / `paddle.CUDAPlace(0)`
- XPU: `paddle.set_device("xpu")` / `paddle.XPUPlace(0)`

Ocean 的 `XPUAccelerator` 已封装此差异，用户只需 `accelerator="xpu"`。

### 监控

```bash
xpu_smi   # 类似 nvidia-smi
```

### 注意事项

1. `paddle.device.get_all_custom_device_type()` **不包含** `"xpu"`
2. `paddle.stft()` 产生的 complex128 在 XPU 上跑 `abs()` 会失败 — Ocean 的 compat 测试已自动 skip
3. 环境变量必须在 Python 进程启动前设置
