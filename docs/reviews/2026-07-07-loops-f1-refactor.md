# Ocean `loops/` F1 重构落实报告

- **日期**：2026-07-07
- **范围**：合并两套并行的训练循环（根因 F1），连带修复 F5 / F4 / F12，并处理 PaddlePaddle 与参考实现的学习率调度差异
- **对照参考**：`lightning-ref/src/lightning/pytorch/loops/`（fe6b1cc）
- **主分支状态**：`master @ 06f397f`（本次开工前已 pull，含近期训练踩坑修复 #28~#31）
- **前置报告**：`docs/reviews/2026-07-06-loops-review.md`（F1~F19）

## TL;DR

- 把内联在 `_FitLoop` 里的整套 batch 循环下沉到 `_TrainingEpochLoop`，激活此前的死代码，形成 `_FitLoop（epoch）→ _TrainingEpochLoop（batch）→ _AutomaticOptimization / _ManualOptimization（step）` 三层结构，对照参考实现。
- 顺带消除的下游 bug：**F5**（跨 epoch 梯度累积、epoch 末强制 flush）、**F4**（手动优化假 step 计数）、**F12**（`enumerate(start=skip)` 假跳过）。
- 因地制宜处理 PaddlePaddle 差异：optimizer 持有 scheduler（与参考实现相反），补充**未绑定告警**与 **ReduceOnPlateau 传参**。
- 回归：改动前后 loop 相关子集 **79 passed / 1 skipped 逐字匹配**；新增 `tests/test_loop_refactor.py`（10 例）；全量 **131 passed / 19 skipped**（19 为 CPU 环境跳过的多卡用例）。

## 修改的文件

| 文件 | 改动 |
|---|---|
| `ocean/loops/fit_loop.py` | 瘦身为纯 epoch 编排：`on_train_start/end`、epoch 钩子、epoch-interval 调度、委派 `epoch_loop.run()`；删除内联 batch 循环 / `opt_acc` / epoch 末 flush / `enumerate(start=skip)`；新增 checkpoint legacy schema 兼容 |
| `ocean/loops/training_epoch_loop.py` | 重写为真正的 batch 循环：batch 钩子、累积判定（基于 batch progress）、step-interval 调度、mid-epoch 验证、`log_every_n_steps`，dispatch 到优化子循环 |
| `ocean/loops/optimization/automatic.py` | 补全 `run()`：完整 callback 钩子、`norm`/`value` 两种 clip、累积决策由 epoch 层传入、step 走 `OceanOptimizer` wrapper（单一计数来源） |
| `ocean/loops/optimization/manual.py` | 纯透传；不再伪造 `_optimizer_step`（F4） |
| `ocean/core/optimizer.py` | `_warn_unbound_schedulers`：scheduler 未作为 `learning_rate` 绑进 optimizer 时告警 |
| `ocean/model.py` | `lr_scheduler_step` 补 `ReduceOnPlateau` 分支（按类型传 metric） |
| `tests/test_loop_refactor.py` | 新增 10 例，覆盖本次全部修复点（即对应 CI） |

## 逐条落实

### F1. 死代码激活 → 三层结构

- **前**：`_TrainingEpochLoop` + `optimization/` 仅被 import、从不实例化；`_FitLoop.run()` 内联重写 batch 循环（227 行）。
- **后**：`_FitLoop.__init__` 持有 `self.epoch_loop = _TrainingEpochLoop(trainer)`；`_FitLoop.run()` 只做 epoch 层，`self.epoch_loop.run()` 承接单 epoch 的 batch 循环。职责划分对照参考实现（epoch 层 / batch 层 / 优化层）。
- **迭代方式**：保留直接 `enumerate(iter(train_loader))`（规避 PaddlePaddle 共享内存问题），`is_last_batch` 由缓存的 `max_batches` 推导，而非引入预取 fetcher。

### F5. 梯度累积用 batch progress，删除 epoch 末 flush

- 累积判定：`current.ready % accumulate == 0` 或 `is_last_batch`（末批强制 step），取代局部 `opt_acc`。
- 计数落在 `batch_progress` 上，随 checkpoint 存取，跨 epoch / 断点续训不再错位；删除 epoch 末的 leftover flush 块，末批 step 在正常循环内完成，无重复 step。
- 覆盖：`test_accumulation_even_division` / `_forces_step_on_last_batch` / `_across_two_epochs`。

### F4. 手动优化不再伪造 step 计数

- `_ManualOptimization.run` 只调 `training_step`；`_optimizer_step` 仅由用户 step 包装后的 `OceanOptimizer`（`_on_after_step` → `_advance_optimizer_step`）推进。
- `dataloader_step` 仍每 batch 递增以驱动 `max_steps`，避免手动模式无法停止。
- 覆盖：`test_manual_mode_does_not_fake_optimizer_step`。

### F12. 删除假跳过

- 移除 `enumerate(data_iter, start=skip)`，续训不再给 batch 打假标签、也不重放。断点进度经 `batch_progress` state_dict 恢复。

### PaddlePaddle 学习率调度差异（因地制宜）

参考实现中 scheduler 持有 optimizer 并反写其 lr；PaddlePaddle 相反——optimizer 持有 scheduler，`scheduler.step()` 仅推进内部状态，`optimizer.step()` 时读取当前 lr。据此：

1. **未绑定告警**：`_warn_unbound_schedulers` 检查 scheduler 是否为某 optimizer 的 `_learning_rate`；否则 `scheduler.step()` 静默无效，给出 `UserWarning`。
2. **ReduceOnPlateau**：其 `step(metric)` 签名与普通 scheduler 不同，`lr_scheduler_step` 按类型分支传入 monitor 指标。
3. step / epoch 两种 interval 的调度**调用点**对照参考（step 在 batch 后、epoch 在 epoch 末），**语义**按 Paddle（纯 `scheduler.step()` 推进）。
- 覆盖：`test_unbound_scheduler_warns` / `test_bound_scheduler_does_not_warn` / `test_reduce_on_plateau_steps_with_metric`。

## Checkpoint 兼容

`batch_progress` 从 `_FitLoop` 移入 `epoch_loop` 后，`fit_loop.state_dict()` 由 `{"batch_progress": ...}` 变为 `{"epoch_loop": {"batch_progress": ...}}`。`_FitLoop.load_state_dict` 探测旧 schema（顶层 `batch_progress`）并路由到 `epoch_loop.batch_progress`，旧 checkpoint 续训不丢进度。覆盖：`test_loop_state_dict_round_trips_through_epoch_loop` / `test_legacy_checkpoint_schema_loads`。

## 验证

- 跑法（本地环境有同名 `tests` 包污染）：`python -m pytest ... --import-mode=importlib -o addopts=""`
- loop 相关子集：改动前后均 **79 passed / 1 skipped**，逐字匹配，零回归
- 新增用例：`tests/test_loop_refactor.py` **10 passed**
- 全量（含 api_coverage / metrics / multi_gpu）：**131 passed / 19 skipped**
- lint：`ruff check` 全过；`ruff format --check` 全过

## 本次未处理（留后续）

- **F3**：`val_check_interval` 的 float/str/时间三态（属 trainer 层，独立改动）
- **F9**：`_metrics_buffer` 按 stage 分离（属 `_LoggerConnector` review）
- **F6 / F8**：PrecisionPlugin 统一 clip、`_EvaluationLoop` 的 limit_batches / sanity
- **F7 / F10~F19**：中低危项
