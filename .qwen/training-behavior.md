# Ocean Training Behavior — 训练引擎行为详解

本文档详细描述 ocean Trainer 的训练行为，确保所有用户自定义模型的行为对齐 PyTorch Lightning。Codec/如来唱项目的所有修复经验均在此记录。

---

## 核心训练流程

```
trainer.fit(model, train_dataloader, val_dataloader)
  │
  ├── _fit_impl()
  │   ├── _data_connector.attach_data()    # 挂载数据
  │   ├── strategy.connect(model)
  │   ├── _call_setup_hook()
  │   ├── model.to(device)                  # 移动到设备
  │   ├── _resolve_optimizers()
  │   ├── ckpt restore (可选)
  │   ├── _sanity_check()                   # 验证前的 sanity check
  │   └── fit_loop.run()                    # 训练主循环
  │       ├── 初始日志 flush (step 0)
  │       ├── epoch 循环
  │       │   ├── batch 循环
  │       │   │   ├── training_step()
  │       │   │   ├── 定期日志 flush（每 log_every_n_steps）
  │       │   │   ├── 验证检查（每 val_check_interval）
  │       │   │   └── max_steps 停止检查
  │       │   └── epoch 结束验证
  │       └── 停止条件检查
  └── on_fit_end()
```

---

## Sanity Check 行为

### 代码位置

`ocean/trainer/__init__.py` — `_sanity_check()`

### 执行流程

```
_sanity_check(model, device):
    1. reset_validation_metrics()         # 清空已验证 metrics（重要！）
    2. self.sanity_checking = True
    3. model.eval()
    4. on_sanity_check_start()            # 回调
    5. on_validation_start()              # 模型钩子 + 回调
    6. for each dataloader:
         for batch_idx in range(num_sanity_val_steps):
             model.validation_step(batch, batch_idx)
             # 注意：validation_step 内通过 logger.experiment.add_scalar()
             # 直接写入 VDL 是允许的（DiffSinger 风格）
    7. on_validation_epoch_end()          # 模型钩子 + 回调
    8. on_sanity_check_end()
    9. self.sanity_checking = False
    10. reset_validation_metrics()        # 再次清空 val metrics！
    11. model.train()
```

### 关键点

1. **sanity check 期间** `self.sanity_checking = True`，模型代码可以通过 `model._trainer.sanity_checking` 判断
2. **Lightning 行为**：sanity check 期间跳过 `update_eval_step_metrics()`，即不通过 `_logger_connector` 写入指标。但 `validation_step` 中直接通过 `logger.experiment.add_scalar()` 的写入仍然触发
3. **前后两次 `reset_validation_metrics()`**：防止 val metrics 污染训练 step 0 的日志 flush
4. **`on_validation_start/epoch_end` 钩子**：必须在 sanity check 中调用，确保模型可以正确地初始化/累积/写入验证结果

### 验证模式中写入 VDL 的推荐方式

```python
def validation_step(self, batch, batch_idx):
    # 1. 累积 loss（所有 batch）
    self._val_losses.append(loss.item())

    # 2. 可视化（DiffSinger 风格：直接写入 VDL）
    if batch_idx < self.config.get("num_valid_plots", 2):
        logger.experiment.add_audio(f"val/audio_{batch_idx}", ...)
        logger.experiment.add_image(f"val/mel_{batch_idx}", ...)

def on_validation_epoch_end(self):
    # 3. 写入 mean loss（一次，避免重复）
    if self._val_losses:
        mean_loss = sum(self._val_losses) / len(self._val_losses)
        logger.experiment.add_scalar("loss/val", mean_loss, self.global_step)
```

---

## 训练循环（FitLoop）

### 代码位置

`ocean/loops/fit_loop.py` — `_FitLoop.run()`

### 初始日志 flush

```python
# 在 epoch 循环开始前
if trainer.log_every_n_steps > 0:
    trainer._logger_connector.log_metrics(trainer.logged_metrics, step=0)
```

这一步确保 step 0 的初始 metrics 被写入 VDL。此时 `_logged_metrics` 可能还为空（如果没有通过 `log()` 记录任何值），但 flush 调用本身不会产生问题。

### 定期日志 flush

```python
# 在 batch 中的 training_step 之后
if trainer.dataloader_step % max(1, trainer.log_every_n_steps) == 0:
    trainer._logger_connector.log_metrics(trainer.logged_metrics, trainer.dataloader_step)
```

- 使用 `dataloader_step`（**递增前**的值），确保 step 0 (0 % N == 0) 触发
- 在 manual_optimization 模式下，`dataloader_step` 在 training_step 结束后才递增

### 自动 vs 手动优化

**自动优化** (`automatic_optimization=True`)：Trainer 内部自动执行 backward、梯度裁剪、optimizer.step()

```python
if model.automatic_optimization:
    loss.backward()
    opt_acc += 1
    if opt_acc >= accumulate_grad_batches:
        clip_grad_norm_()
        optimizer.step()
        dataloader_step += 1
```

**手动优化** (`automatic_optimization=False`)：模型在 training_step 内部自行处理 backward 和 optimizer.step()

```python
else:
    dataloader_step += 1
    optimizer_step += 1
```

### 验证检查点

```python
# 基于 step 的验证
if trainer._should_check_val_step(trainer.dataloader_step):
    self._run_validation()
```

`_should_check_val_step(step)` 检查 `step % val_check_interval == 0`

---

## 验证流程（FitLoop._run_validation）

### 执行流程

```
_run_validation():
    1. model.on_validation_model_eval()
    2. on_validation_start()              # 模型钩子 + 回调
    3. on_validation_epoch_start()
    4. for each dataloader:
         for batch_idx in range(limit_val_batches):
             model.validation_step(batch, batch_idx)
    5. trainer._compute_epoch_metrics()    # 计算 epoch 级 metrics
    6. trainer._logger_connector.log_metrics(trainer.logged_metrics, step)
       # ⚠ 这里 flush 验证 metrics 到 VDL（step = dataloader_step）
    7. on_validation_epoch_end()          # 模型钩子 + 回调
    8. on_validation_end()                # 模型钩子 + 回调
    9. reset_validation_metrics()         # 清空 val metrics
    10. model.on_validation_model_train()
```

### 验证 metrics flush

在 `_run_validation()` 中，步骤 6 会通过 `_logger_connector.log_metrics()` 将 `trainer.logged_metrics` 中的 value 写入 logger。这意味着：

- 如果模型在 `validation_step` 中调用了 `self.log("loss/val", ...)`，该值会出现在 `_logged_metrics` 中
- 在 `on_validation_epoch_end` 后，`reset_validation_metrics()` 会清空所有 val metrics

**实践建议**：如果使用 DiffSinger 风格（`validation_step` 中累积 + `on_validation_epoch_end` 中直接写 VDL），则避免在 `validation_step` 中调用 `self.log("loss/val", ...)`，以免被 `log_metrics` flush 和 VDL 直接写入造成重复。

---

## Logger Connector

### 代码位置

`ocean/trainer/connectors/__init__.py` — `_LoggerConnector`

### 核心方法

#### `log_metrics(metrics, step=None)`

将 metrics dict 分发给所有 logger 的 `log_metrics()` 方法：

```python
def log_metrics(self, metrics, step=None):
    for lg in self.trainer.loggers:
        if hasattr(lg, 'log_metrics'):
            lg.log_metrics(metrics, step)
```

- VisualDLLogger 的 `log_metrics()` 调用 `experiment.add_scalar(k, v, step)`
- step 为 None 时跳过（VisualDLLogger 实现）

#### `reset_validation_metrics()`

全量清空 `_logged_metrics` 和 `_metrics_buffer`：

```python
def reset_validation_metrics(self):
    self._logged_metrics.clear()
    self._metrics_buffer.clear()
```

**为什么需要清空 `_logged_metrics`**：如果不清空，`loss/val` 等键会残留在 `_logged_metrics` 中，下一次 `log_metrics()` 调用时会再次写入，导致重复记录。

#### `log_metric_value(name, value, prog_bar=False)`

单指标写入（从 model 的 `log()` 和 `log_dict()` 调用）：

```python
def log_metric_value(self, name, value, prog_bar=False):
    self._callback_metrics[name] = value
    self._logged_metrics[name] = value
    if prog_bar:
        self._progress_bar_metrics[name] = value
    if name not in self._metrics_buffer:
        self._metrics_buffer[name] = []
    self._metrics_buffer[name].append(value)
```

---

## VisualDLLogger

### 代码位置

`ocean/loggers/visualdl.py` — `VisualDLLogger`

### 日志目录结构

```
{save_dir}/{name}/{version}/
```

- `save_dir`：配置中的 `log_dir`（如来唱：`logs/codec`）
- `name`：固定 `"ocean_logs"`
- `version`：`"latest"`（如来唱配置）或自动递增版号

### version="latest" 模式

```python
VisualDLLogger(save_dir, version="latest")
```

- `self._version = "latest"`（字符串）
- `self.log_dir = "{save_dir}/ocean_logs/latest"`
- 每次运行覆盖写入，不产生 version_N 子目录
- **注意**：使用 `"latest"` 时 `_get_next_version()` 不会被调用

### log_metrics 行为

```python
def log_metrics(self, metrics, step=None):
    if step is None:
        return
    for k, v in metrics.items():
        key = f"{self._prefix}/{k}" if self._prefix else k
        if hasattr(v, "item"):
            v = v.item()
        self.experiment.add_scalar(key, float(v), step)
```

- **step=None 时跳过**：避免无 step 信息的 metrics 写入
- **不会抛出异常**：如果 VisualDL 未安装，`_create_experiment()` 返回 dummy writer

---

## Model 钩子

### 代码位置

`ocean/model.py`

### 关键属性

| 属性 | 说明 |
|------|------|
| `self.global_step` | 别名 `dataloader_step`（来自 trainer） |
| `self.logger` | 第一个 logger |
| `self.automatic_optimization` | 是否自动优化 |
| `self.log()` | 单指标日志（调 trainer._log_metric） |
| `self.log_dict()` | 多指标日志 |

### 30+ 生命周期钩子

```python
# 训练
on_fit_start / on_fit_end
on_train_start / on_train_end
on_train_epoch_start / on_train_epoch_end
on_train_batch_start / on_train_batch_end

# 验证
on_validation_start / on_validation_end
on_validation_epoch_start / on_validation_epoch_end
on_validation_batch_start / on_validation_batch_end

# 反向传播
on_before_backward / on_after_backward
on_before_optimizer_step / on_before_zero_grad
```

### compile() 与 CINN

```python
def compile(self, full_graph=False, input_spec=None):
    """应用 paddle.jit.to_static 加速训练。"""
    # 同时编译 forward 和 training_step
    # 返回 self
```

- 编译后 `forward` 和 `training_step` 运行在静态图模式下
- 使用 CINN 编译（需 Paddle 3.x 以上版本）
- **预热机制**：在 `Trainer.fit()` 前用 dummy input 执行一次 forward，触发编译

---

## SOT KeyError 'self' 补丁

### 代码位置

`ocean/_compat/sot.py` — `patch_sot()`

### 自动调用

`ocean/__init__.py` 导入时自动调用 `patch_sot()`。

### 修复原理

当 `model.compile()` 同时编译多个方法（如 `forward` + `training_step`）时，SOT 模拟字节码时 `VariableFactory.from_value` 对 `StaticFunction` 创建了错误类型的变量。补丁在运行时替换 `VariableFactory.mapping_str_func["UserDefinedFunctionVariable"]`，使其检测 bound `StaticFunction` → 创建 `MethodVariable`。

### 触发条件

只有 `self.method()` 调用（`LOAD_METHOD` + `CALL_METHOD`）才会触发。纯属性读取不受影响。

---

## 配置加载（DiffSinger 风格）

```python
# 支持 base_config 继承
def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if "base_config" in config:
        for bp in config["base_config"]:
            with open(bp) as f:
                base = yaml.safe_load(f)
            merged = base.copy()
            merged.update(config)
            config = merged
    return config
```

### 环境变量提前设置

```python
# 在 import paddle 之前设置
os.environ["GLOG_minloglevel"] = str(config["glog_level"])
```

---

## 常见问题排查

### 1. VDL 中 loss/val 有重复记录

**现象**：同一个 step 出现两条 loss/val（两个不同值）。

**排查**：
- `validation_step` 是否直接写入 VDL？→ 改为累积 + on_validation_epoch_end 写入
- `_run_validation` 中是否有 `log_metrics` flush？→ 检查是否有重复 flush
- `reset_validation_metrics` 是否被调用？→ 验证后必须清空

### 2. VDL 中 loss/val 缺失或为空

**现象**：step 0 没有 loss/val 记录。

**排查**：
- `on_validation_epoch_end` 是否写入 VDL？→ 确认 `logger.experiment.add_scalar()` 被调用
- sanity check 是否调用了 `on_validation_start/epoch_end` 钩子？
- 模型 `on_validation_start` 是否正确重置了 `_val_losses`？

### 3. train/ 下 step 0 有多个 Logger 记录

**现象**：train/ 下同一指标出现两条 step 0 记录。

**排查**：
- sanity check 是否在 flush `_logged_metrics`？→ 应通过 `reset_validation_metrics` 清空
- fit_loop 的初始 flush 是否不需要？→ 需要保留（Lightning 行为）

### 4. CINN 编译导致 sanity check 卡住

**现象**：sanity check 的第一个 `validation_step` 卡住数分钟。

**排查**：
- 预热步骤是否在 `Trainer.fit()` 之前执行？
- 预热使用的 dummy input shape 是否与实际一致？

---

## 与 Lightning 的关键差异

| 方面 | Lightning | Ocean |
|------|-----------|-------|
| `_logged_metrics` 管理 | val/train 分离 | 共享字典，需手动 reset |
| `validation_step` 写入 | 通过 `log_metrics` 间接 | 可直接写 VDL (DiffSinger 风格) |
| CSVLogger 默认 | 否 | 是（当 logger=True/None） |
| `_sanity_check` 钩子 | 不调用 `on_validation_*` | 调用 `on_validation_start/epoch_end` |
| `compile()` | 无 | 使用 `paddle.jit.to_static` |
| SOT 补丁 | 不适用 | 需要 `patch_sot()` |
