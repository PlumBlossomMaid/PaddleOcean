# Ocean `loops/` 行级 Review 报告

- **日期**：2026-07-06
- **范围**：`ocean/loops/fit_loop.py`、`training_epoch_loop.py`、`evaluation_loop.py`、`prediction_loop.py`、`loop.py`、`fetchers.py`、`progress.py`、`utilities.py`、`optimization/`
- **对照参考**：`lightning-ref/src/lightning/pytorch/loops/`（fe6b1cc）
- **主分支状态**：`master @ aab50df`

## TL;DR

- 发现 **1 个严重架构问题** + **6 个高危训练 bug** + 若干中低危问题
- 核心根因：`_TrainingEpochLoop` 存在但**没有任何地方在用**，`FitLoop` 自己重写了一遍训练循环，两套代码不一致
- 近 30 个 commit 大多在打补丁，先解决 F1（合并两���循环）能顺带消掉多个下游 bug

## 🔴 严重（会导致训练结果错误）

### F1. `_TrainingEpochLoop` 是死代码，`FitLoop` 内联重写训练循环

**位置**：
- `ocean/loops/fit_loop.py:66-137` — 内联训练循环
- `ocean/loops/training_epoch_loop.py`（整个文件，95 行）
- 参考：`fit_loop.py` `advance()`：只做 `self.epoch_loop.run(self._data_fetcher)`

**证据**：`grep -rn "epoch_loop" ocean/` 只在 `__init__.py` 里 re-export；`FitLoop` 里搜不到 `epoch_loop` / `training_epoch_loop`。而 `fit_loop.py:83-137` 手写了 batch 循环、梯度累积、优化器 step、val 触发。

**后果**：
- `training_epoch_loop.py` 里的 `_AutomaticOptimization.run()`、`update_lr_schedulers()`、`_should_check_val_fx()` 全部无效
- `automatic_optimization = False` 时，`ManualOptimization.run()` **永远不会被执行**（`fit_loop.py:118-123` 只是把 `_dataloader_step += 1` 和 `_optimizer_step += 1`）
- LR scheduler step 完全没有被调用（见 F2）

**建议**：
1. **短期**：删除 `training_epoch_loop.py` + `optimization/` 目录，避免误以为它们生效
2. **正确**：`FitLoop.run()` 只保留 epoch 层，把 batch 循环移到 `_TrainingEpochLoop`（这也解释了为什么 Ocean fit_loop 只有 209 行 vs 参考实现 555 行还漏了一堆功能）

---

### F2. LR scheduler 完全不会被 step

**位置**：`ocean/loops/fit_loop.py:66-137`（`FitLoop.run()` batch 循环）

**证据**：全文搜 `scheduler` 只出现在 checkpoint dump/restore（`trainer/__init__.py:660`、`connectors/__init__.py:275`）和 `_lr_schedulers = []`（`trainer/__init__.py:200`）。**训练主循环里没有任何地方调 `scheduler.step()`**。

**后果**：用户 `configure_optimizers` 返回 `{"optimizer": ..., "lr_scheduler": ...}` 会保存进 `self._lr_schedulers`、断点会存/恢复，但训练时学习率永远是初始值。属于**无声正确性 bug**（能跑，结果不对）。

**建议**：
- `fit_loop.py:126` 的 `opt_acc = 0` 之后（每次 `optimizer.step()` 之后）加入按 `interval="step"` 的调度
- `fit_loop.py:158` `_call_module_hook(..., "on_train_epoch_end")` 之前加入按 `interval="epoch"` 的调度
- 区分 plateau/非-plateau，plateau 依赖 metric，需在 val 之后 step

---

### F3. `_should_check_val_step` 只在 step 里检查，last-batch / max-epoch / stop 时不检查

**位置**：`ocean/trainer/__init__.py:775-784` + `ocean/loops/fit_loop.py:131`

**对照参考实现的判定顺序**：
1. `_should_check_val_epoch`（是否本 epoch 该 val，`check_val_every_n_epoch`）
2. `is_last_batch and (is_infinite_dataset or DataLoaderIterDataFetcher)`
3. `trainer.should_stop and _can_stop_early` — 提前停止时也要 val
4. `_val_check_time_interval` — 时间触发
5. `val_check_batch % (batch_idx+1) == 0`

Ocean 只保留了第 5 条，且 `val_check_interval` 的 int/float/str 三态没处理：
```python
# trainer/__init__.py:782
if not isinstance(val_interval, int) or val_interval <= 0:
    return False
```

**后果**：
- `check_val_every_n_epoch=2` 无效，Ocean 会每 epoch val
- 迭代式数据集不做 last-batch val
- Early stopping 触发 `should_stop=True` 时不会跑最后一次 val
- `val_check_interval="00:30:00"`（时间）无支持
- `val_check_interval=0.5`（float，表示"每 epoch 一半 val 一次"）被吞

**建议**：`val_check_interval` 的 int/float/str 三态在 `_data_connector` 里预处理成 `val_check_batch: int`，`_should_check_val_step` 重写成对齐参考的五段判定。

---

### F4. 手动优化模式 `_optimizer_step` 计数错误

**位置**：`ocean/loops/fit_loop.py:118-123`
```python
else:                                # automatic_optimization == False
    trainer._dataloader_step += 1
    trainer._optimizer_step += 1     # 用户还没 step 就 +1
    step = trainer.optimizer_step
    if step > 0 and step % max(1, trainer.log_every_n_steps) == 0:
        trainer._logger_connector.log_metrics(trainer.logged_metrics, step)
```

**后果**：
- Ocean 手动模式下每个 batch 无脑把 `_optimizer_step += 1`
- 用户实际调 `self.optimizers.step()` 时**没有**任何计数递增
- log 的 x 轴（`global_step`）跟实际的 optimizer step 数完全脱钩

另外 `fit_loop.py:78-80`：
```python
_call_callback_hooks(trainer, "on_train_batch_start", batch, batch_idx)
skip_flag = model.on_train_batch_start(batch, batch_idx)
if skip_flag == -1:
    continue           # 已 increment_ready，completed 永远赶不上
```

**建议**：
- 删掉 `fit_loop.py:118-123` 的假 step，`_optimizer_step` 只在 `_advance_optimizer_step`（`trainer/__init__.py:518`）里递增，由用户 `optimizer.step()` 触发
- `skip_flag == -1` 时补齐 `increment_completed()` 或用 `increment_by(0, is_last_batch=True)`

---

## 🟠 高危（行为不一致 / 训练中崩）

### F5. 梯度累积计数用 `opt_acc` 局部变量，跨 epoch 丢失、跨 restart 丢失

**位置**：`ocean/loops/fit_loop.py:73`（`opt_acc = 0` 每 epoch 清零）

**后果**：
- Ocean 每 epoch 开头 `opt_acc = 0`；`dataset_size % accumulate_grad_batches != 0` 时，epoch 末的 `if opt_acc > 0:` flush（`fit_loop.py:141-155`）会**每个 epoch 强制 step 一次尾部**
- 某些情况下会导致梯度 magnitude 翻倍
- 断点续训时 `opt_acc` 不在 state_dict 里，恢复后从 0 开始也会产生一次错位 step

**建议**：删掉 `opt_acc`，改为 `batch_progress.current.ready % accumulate_grad_batches == 0`；epoch 末尾的强制 flush 逻辑（`fit_loop.py:141-155`）删除。

---

### F6. gradient clip 分支拼错 + 跳过精度插件

**位置**：`ocean/loops/fit_loop.py:99-103` 和 `144-148`

**问题**：
1. `paddle.nn.utils.clip_grad_value_` 在部分 Paddle 版本下不存在（Paddle 用 `ClipGradByValue` 的类式接口），需 fallback
2. `automatic.py:59` 的 clip 硬编码用 `clip_grad_norm_`，没判 `algorithm`（当前是死代码，一旦按 F1 修好会暴露）
3. Ocean 直接调 `paddle.nn.utils.*` 绕过了 `PrecisionPlugin`；AMP 场景下会 clip 到还没 unscale 的梯度上

**建议**：抽 `_clip_gradients(optimizer)` 到 `PrecisionPlugin`，`fit_loop` 只调 `trainer.precision_plugin.clip_gradients(optimizer, clip_val, algorithm)`。

---

### F7. `EvaluationLoop` 里 `device = trainer._resolve_device()` 放在双层 for 里

**位置**：`ocean/loops/evaluation_loop.py:74`

```python
for dl_idx, dataloader in enumerate(dataloaders):
    for batch_idx, batch in enumerate(dataloader):
        device = trainer._resolve_device()   # 每个 batch 都 resolve
```

**后果**：热路径下白开销。`fit_loop.py:60` 已经做对了（提到循环外）。

**建议**：把 `device = trainer._resolve_device()` 提到 `with paddle.no_grad():` 之前。

---

### F8. `EvaluationLoop.run` 无视 sanity_check / 无视 `limit_val_batches`

**位置**：`ocean/loops/evaluation_loop.py:72-83`

**后果**：`trainer.validate(...)` 或后续走 `_EvaluationLoop.run` 的路径会跑完整个 val set，`limit_val_batches=0.1` 或 `fast_dev_run=True` 都不生效。

（注意：`_FitLoop._run_validation` 是自己写的，那里有 `_should_limit_batches`；`trainer.validate()` 走 `validate_loop.run`，才进 `evaluation_loop.py`）

**建议**：`evaluation_loop.py:75` 的 `for batch_idx` 前加 `if trainer._should_limit_batches(batch_idx, "val"): break`。sanity_check 目前是 trainer 里手写的独立循环（`trainer/__init__.py:795-836`），不走 `evaluation_loop`。

---

### F9. Val 循环里 `_compute_epoch_metrics` 之后又清 `_metrics_buffer`，训练 metrics 会丢

**位置**：
- `ocean/loops/fit_loop.py:189` `trainer._compute_epoch_metrics()`（val 结束）
- `ocean/loops/fit_loop.py:158` `trainer._compute_epoch_metrics()`（train epoch 结束）
- `ocean/trainer/__init__.py:690` `self._log_metrics_buffer.clear()`

**流程**：mid-epoch val（如 `val_check_interval=100`）触发 `_run_validation` → `_compute_epoch_metrics` → clear buffer → 训练继续。此时训练 metrics buffer 空，`on_train_epoch_end` 的 `_compute_epoch_metrics` 只包含 val 之后的训练 batch，前 100 个 batch 的训练 loss 均值全丢。

**建议**：`_log_metrics_buffer` 拆成 `_train_metrics_buffer` / `_val_metrics_buffer` / `_test_metrics_buffer`，`_compute_epoch_metrics` 只清对应 stage 的 buffer。参考实现是通过 `_ResultCollection(training=True/False)` 天然分离。

---

## 🟡 中危

### F10. `EvaluationLoop.run` 用 `inspect.signature` 判断 `dataloader_idx` 支持——太贵

**位置**：`ocean/loops/evaluation_loop.py:79-83`

**建议**：`on_run_start` 里 cache 结果。

---

### F11. `FitLoop.run` 里 `_restarting = False` 时机在 epoch 循环内部而非 `on_iteration_done`

**位置**：`ocean/loops/fit_loop.py:167` (`self._restarting = False`)

**后果**：epoch 中间抛异常（如 SIGTERM），`_restarting` 不会被清，下次同一进程再 `fit` 时残留。因为 Ocean fit 结束才拆 flag（`fit_loop.py:180`），场景较边缘。

---

### F12. 断点续训的 skip 逻辑基于 `enumerate(data_iter, start=skip)` 但 Paddle DataLoader 不支持跳过

**位置**：`ocean/loops/fit_loop.py:71-72`

```python
data_iter = iter(train_loader)
skip = self.batch_progress.current.ready
for batch_idx, batch in enumerate(data_iter, start=skip):
```

**问题**：`enumerate(x, start=k)` 只是让 `batch_idx` 从 k 开始，**不会真的跳过前 k 个 batch**，train loader 从头出 batch 但被打上 `batch_idx=k, k+1, ...`。断点续训语义错。

**验证**：`git log` 里 `55f4f63 remove skip loop on checkpoint resume`——历史上有过一版 skip 逻辑被移除，这行遗留下来看起来在装样子。

**建议**：改成 `start=0`，明确"断点续训不跳 batch"，跟 commit 55f4f63 的意图对齐。

---

### F13. `on_before_zero_grad` 语义位置差异

**位置**：`ocean/loops/fit_loop.py:107-109 / 149-151`

Ocean 在 `optimizer.step()` 之后、`clear_grad()` 之前 fire，参考实现是在 `optimizer_step()` 之前 fire。用户在 `on_before_zero_grad` 里读 `param.grad` 时读到的梯度状态不同。不会崩，但语义不对齐。

---

### F14. `_call_callback_hooks` 缺 `monitoring_callbacks` 分阶段调用

**位置**：`ocean/trainer/call.py:5-11`

参考实现的 `_call_callback_hooks` 有 `monitoring_callbacks: Optional[bool] = None` 参数，让 `EarlyStopping/ModelCheckpoint` 这类 monitor callback 在 `LightningModule.on_train_epoch_end` **之后**执行才能看到用户 log 的 metric。Ocean 的 `fit_loop.py:159-160` 是先 model 后 callback，简化版顺序对，但用户如果在 callback 的 `on_train_epoch_end` 里也 `self.log(...)`，会跟 model 后写入的 metric 顺序混乱。

**建议**：短期在 QWEN.md 记录差异；长期复刻两阶段调用。

---

## 🟢 低危 / 代码卫生

### F15. `FitLoop.__init__` 里 `max_epochs or 1000`
**位置**：`ocean/loops/fit_loop.py:20`
默认 1000 是魔法数字。`max_epochs=0`（合法：只 sanity_check 不训练）会被静默替换。

### F16. `_TrainingEpochLoop._should_check_val` 硬编码 `return False`
**位置**：`ocean/loops/training_epoch_loop.py:88`
F1 的死代码副产品。

### F17. `_EvaluationLoop.run` 返回 `[dict(trainer._log_metrics_on_epoch)]`
**位置**：`ocean/loops/evaluation_loop.py:97`
`_log_metrics_on_epoch` 是"当前 epoch 所有 log"，val 和 train 混在一起。`validate()` / `test()` 返回值应只含本次评估结果。

### F18. `_PrefetchDataFetcher.__next__` 状态机的 EOF 判断
**位置**：`ocean/loops/fetchers.py:47-63`
预取第一次拿到即 EOF 时（数据集只有 1 个 batch），下一次 `__next__` 会返回 `None` 而非 raise `StopIteration`。

### F19. `_PredictionLoop.run` return 类型不一致
**位置**：`ocean/loops/prediction_loop.py:14`
signature `-> list[Any]` 但空 dataloader 时 `return [], []`（两个值），正常路径 `return predictions`（一个 list）。

---

## 复核最近 30 个 commit

| commit | 意图 | 与本报告映射 |
|---|---|---|
| `bdf7ac4` fix: populate logged_metrics from _compute_epoch_metrics | 修 F9 症状 | 根因（stage 未分离）还在 |
| `349c05b` fix: add max_batches -> num_training_batches | 补 F1 症状 | 真正 batch 计数在死掉的 epoch_loop 里 |
| `c92c4b7` / `cab135d` reset batch_progress per epoch | F5 同源 | 症状打补丁 |
| `55f4f63` remove skip loop on checkpoint resume | F12 半成品 | 忘删 `enumerate(start=skip)` |
| `8d31c3f` use batch_idx for val check | F3 只修了 1/5 | 缺其他 4 条判定 |

**判断**：这些 commit 都在打补丁，但根因是 **F1**——两套并行的训练循环。先解决 F1，很多下游 bug 会自动消失。

---

## 建议的修复顺序

1. **F1**：合并两套循环，`FitLoop` 瘦身，batch 循环下沉到 `_TrainingEpochLoop`
2. **F2**：LR scheduler step 接入训练主循环
3. **F3**：`val_check_interval` 三态处理 + 五段判定
4. **F9**：`_log_metrics_buffer` 按 stage 拆分
5. **F5 / F12**：梯度累积用 progress 计数、删掉 `enumerate(start=skip)`
6. **F4 / F6 / F8**：手动模式 step 计数、gradient clip 走 PrecisionPlugin、EvaluationLoop 加 limit_batches
7. 中低危顺手清
