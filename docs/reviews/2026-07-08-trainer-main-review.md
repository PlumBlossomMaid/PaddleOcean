# Ocean `trainer/__init__.py` 主体 行级 Review 报告

- **日期**：2026-07-08
- **范围**：`ocean/trainer/__init__.py`（937 行的 Trainer 主体：生命周期 fit/validate/test/predict、状态机、metrics、val 调度、checkpoint、sanity、DDP wrapping）
- **对照参考**：`lightning-ref/.../trainer/trainer.py` + `states.py`
- **主分支状态**：接续分支 `fix/logger-result-collection`（本分支已累计 F9/F3/C1/D1/D2/C2/C3/CB1/CB2/S1/A1/A2/D3 等修复）
- **分级**：🔴 严重（静默错误结果） / 🟠 高危 / 🟡 中危 / 🟢 低危

## 修复进度

### ✅ T1 🔴 running stage 从不设置，`trainer.training/validating/testing/predicting` 恒为 False

**位置**：`trainer/__init__.py` 的四组 property setter（`training`/`validating`/`testing`/`predicting`，读写 `self.state.stage`）+ 各 loop。

**问题**：这四个 property 的 **setter 从未被任何地方调用**，`state.stage` 全程保持 `None`。因此：
- 用户在 `training_step`/`validation_step` 里读 `self.trainer.training` / `.validating`（对标参考的常用 API）永远得到 **False**——静默错误。
- 我在 F9 引入的 `_current_fx()`（按 stage 给 metric 存储 key）也依赖 `state.stage`，此前只靠 collection 交换兜底、fx key 实际全错记成 `training_step.*`（因 collection 已分离才没撞键，但语义错误、脆弱）。

**证据**：`grep` 显示 `state.stage = RunningStage.X` 仅出现在 setter 定义内部，无外部调用；运行时实测 `training_step` 中 `trainer.training == False`。

**修复**（对照参考在 loop 里设置 running stage）：
- `fit_loop.run()`：进入训练循环前 `trainer.training = True`。
- `training_epoch_loop._run_validation()`：mid-epoch val 期间 `trainer.validating = True`，`finally` 里恢复 `trainer.training = True`。
- `evaluation_loop.run()`：standalone validate/test 时 `trainer.state.stage = self.stage`（VALIDATING/TESTING）。
- `_sanity_check()`：设 `state.stage = SANITY_CHECKING`，`finally` 恢复原 stage。
- `_current_fx()`：SANITY_CHECKING 也映射到 `validation_step`。

**验证**：`tests/test_trainer_stage.py`（5 例，含 revert 验证：去掉 stage 设置后训练 flag 测试即失败）。训练/mid-epoch val/standalone validate/test 阶段 flag 全部正确；187 passed 零回归。

### ✅ T2 🟠 `limit_train_batches` 的 float 分数被静默忽略（F3 同族）

**位置**：`fit_loop.py:50`（旧 `_max_batches = len(train_loader)`）+ `training_epoch_loop.py:71` `_should_limit_batches`（只处理 int）。

**问题**：训练循环的有效批数 `_max_batches` 直接取 `len(train_loader)`（原始长度），未经 `_resolve_limit` 调整；而按 `batch_idx >= max_batches` 停止只对 raw 长度生效。`_should_limit_batches` 又只认 int。于是 `limit_train_batches=0.5` 这类**分数被完全忽略，全部 batch 照跑**。与 F3、T1 同族的"只认 int / 漏接线"。

**证据**：实测 `limit_train_batches=0.5`（10 batch dataloader）实际跑满 10 batch（应为 5）。且既有测试 `test_progress_bar.py::test_fit_progress_bar_with_limit_train_batches` 的断言 `total == 8` 附注释 **"Ocean doesn't cap max_batches with limit_train_batches"**——测试本身承认了这个 bug。

**修复**：`fit_loop.run()` 用 `_resolve_limit(train_loader, limit_train_batches)` 的结果设 `_max_batches`（`_resolve_limit` 已正确处理 int 与 float 分数、并 cap 到 total）。同一 resolved 值也喂给 `_setup_val_check_batch`（本就如此）。

**验证**：新增 `tests/test_limit_batches.py`（0.5→5 / 0.3→3 / 1.0→10 / int 3→3 / int 10→10，含 `num_training_batches` 一致性）；修正 `test_progress_bar.py` 那条编码了旧 bug 的断言（`total` 8→3）。193 passed 零回归。

### ✅ T3 🟠 eval 循环忽略 `limit_val_batches` / `limit_test_batches`

**位置**：`evaluation_loop.py` 的 batch 循环（standalone validate/test，**完全无** limit 逻辑）+ `training_epoch_loop.py:192`（mid-epoch val 用 `_should_limit_batches`，只认 int）。

**问题**：
- **T3a**：`_EvaluationLoop.run()` 的 batch 迭代没有任何 limit 检查——standalone `validate()`/`test()` **无视 `limit_val_batches`/`limit_test_batches`**（int 和 float 都被忽略），跑满整个 dataloader。
- **T3b**：mid-epoch val 用 `_should_limit_batches`，float 分数被忽略（同 T2/F3）。

**证据**：实测 `validate(limit_val_batches=0.5)` 在 10-batch loader 上跑满 10。

**修复**：
- `evaluation_loop.run()`：按 stage 取 `limit_{val,test}_batches`，每个 dataloader 用 `_resolve_limit` 算 `max_batches`，`batch_idx >= max_batches` 时 break（支持 int + float）。
- `training_epoch_loop._run_validation()`：同样改用 `_resolve_limit`。
- `limit_val_batches=0` 仍由 `_should_check_val_epoch` 上游禁用整个 val（D2/既有行为不变）。

**验证**：`tests/test_limit_batches.py` 扩展（standalone validate 0.5→5/0.2→2/int3→3/1.0→10；test 同理；mid-epoch val 0.3→3；val=0 禁用）。202 passed 零回归。

### ✅ T4 🟡 `reload_dataloaders_every_n_epochs` 是死配置（accept + validate 但从不生效）

**位置**：`trainer/__init__.py`（仅 __init__ 赋值 + D2 校验 + 传给 on_trainer_init），全代码无任何"重建 dataloader"逻辑。

**问题**：用户可设、且 D2 会校验它是 int≥0，但**训练循环从不据此重建 dataloader**——数据永远不刷新。与 F3/T1/T2/T3 同族的"配了没接线"。

**修复**：`fit_loop` 新增 `_reload_train_dataloader_if_needed()`，每 epoch 开头判断：有 datamodule 且 `current_epoch>0 且 current_epoch % n == 0` → 重跑 `datamodule.setup("fit")` + `train_dataloader()`，并重算有效批数。raw dataloader（无 datamodule）无源可重建，跳过。

**验证**：`tests/test_reload_dataloaders.py`（datamodule 每次 reload 返回更大数据集：reload=1 → 每 epoch 批数 [2,4,4]；reload=0 → [2,2,2]；reload=2 → [2,2,4,4]）。含 revert 验证。205 passed 零回归。

### 小结：Trainer 主体的"配了没生效"家族
T1（running stage）/T2（train float limit）/T3（eval limit）/T4（reload dataloaders）+ 此前 loops 的 F3（val_check_interval）——**同一模式**：参数在 `__init__` 存了、有的还加了校验，但循环里从未真正据此执行。这是 AI 生成框架的典型欠账，建议后续对所有 Trainer 构造参数做一遍"是否真被消费"的系统排查。已确认**已生效**的：`gradient_clip_val`（automatic.py:74）、`log_every_n_steps`（training_epoch_loop:160）、`max_steps`（_should_stop + epoch loop）、`accumulate_grad_batches`（F1 已修）。

### ✅ T5 🟠 `ddp_spawn` 子进程丢弃大部分训练配置

**位置**：`fit()` 的 `_spawned_fit`（子进程重建 Trainer）。

**问题**：子进程用手写的一小撮参数重建 Trainer，**静默丢弃** `max_steps`/`min_epochs`/`limit_*_batches`/`val_check_interval`/`check_val_every_n_epoch`/`gradient_clip_val`/`gradient_clip_algorithm`/`accumulate_grad_batches`/`num_sanity_val_steps`/`reload_dataloaders_every_n_epochs`/`detect_anomaly`——全部回落默认。即 `ddp_spawn` 下梯度裁剪失效、累积重置为 1、batch 限制丢失、val 调度错乱。
- **附带证据**：原代码用了 `self.enable_progress_bar`，但该名并非 Trainer 属性，重建时会直接 `AttributeError`——说明 `ddp_spawn` 这条路径此前基本没被真正跑过。

**修复**：抽出可测的 `_spawn_trainer_kwargs(nprocs)` 转发全部训练配置；`enable_progress_bar`/`enable_checkpointing` 从实际 callbacks 推断；删 spawn 块里未用的 `_AcceleratorConnector` 死 import。守卫 `tests/test_ddp_spawn_config.py`（3 例）。真实多进程 spawn 本 CPU 环境无法跑，仅验证转发配置。208 passed 零回归。

### 🔄 待查（本文件后续 review 计划）
- `_fit_impl` 中 DDP wrapping 后训练循环调 `ddp_model.training_step`→内层 forward，**可能绕过 DataParallel.forward 导致梯度不同步**（经典 DDP 陷阱，需多卡环境验证，本 CPU 环境无法确认，勿盲改）；同处 `_resolve_optimizers(model)` 传的是原始 model（DataParallel 共享参数，初判无碍）。
- eval 进度条 total 未反映 limit（cosmetic）。
- fit 结束后 `state.stage` 未清（低危）。
