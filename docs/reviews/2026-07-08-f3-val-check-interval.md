# Ocean F3 修复：验证触发调度（val_check_interval 三态 + epoch 门控）

- **日期**：2026-07-08
- **起因**：loops review 的 **F3**（`val_check_interval` 只支持 int，float/时间态被吞；`check_val_every_n_epoch`/last-batch/should_stop 未实现）。见 `2026-07-06-loops-review.md`。
- **对照参考**：`loops/fit_loop.py`（`val_check_batch` 计算）+ `loops/training_epoch_loop.py`（`_should_check_val_epoch` / `_should_check_val_fx`）
- **主分支状态**：接续 F9 分支 `fix/logger-result-collection`

## 暴露出的真实严重度

F3 不只是"float 被吞"。查证发现：

- `val_check_interval` **默认 `1.0`（float）**，而旧 `_should_check_val_step` 只认 int → 永远返回 False；
- 旧 epoch 级 `_should_check_val()` 是**死代码**（零调用），fit 里也没有 epoch 末验证调用。

**结论：默认配置下 `fit` 从不触发验证**（此前 F9 工作中修 `test_validation_logging` 时首次发现——那些用例断言的其实是 sanity-check 泄漏的状态）。这是静默行为缺陷，影响所有默认用户。

## 做了什么（对齐参考）

### `Trainer._setup_val_check_batch(max_batches)`（新增，fit 开始时调用一次）
把 `val_check_interval` 解析成本轮调度：
- **int**：每 N 个训练 batch 验证；若 `N > max_batches` 且 `check_val_every_n_epoch is not None` → 抛 `ValueError`（提示可设 `check_val_every_n_epoch=None` 或 `limit_val_batches=0`）。
- **float**：epoch 分数，`val_check_batch = max(1, int(max_batches * interval))`；`1.0` → epoch 末验证一次。
- **时间态**（`str "DD:HH:MM:SS"` / `timedelta` / `dict`）：复用现成 `ocean/callbacks/timer.py::_parse_duration` 解析为秒，存 `_val_check_time_interval`，按 wall-clock 触发。
- `max_batches` 用 **limit 调整后**的有效批数（`_resolve_limit(train_loader, limit_train_batches)`），使 `1.0` 落在有效末批。

### `_should_check_val_epoch()`（替换死代码 `_should_check_val`）
epoch 级门控 = 有 val_dataloaders 且 `limit_val_batches != 0` 且 `(check_val_every_n_epoch is None or (current_epoch+1) % n == 0)`。`+1` 语义对齐参考（旧死代码用 `current_epoch % n`，语义相差一个 epoch）。

### `_should_check_val_step(batch_idx)`（重写）
先过 epoch 门控 → 时间态按 wall-clock → 否则按 `val_check_batch` 取模。`check_val_every_n_epoch is None` 时用**跨 epoch 全局** batch 计数（`current_epoch*max_batches+batch_idx`），否则用 epoch 内 `batch_idx`（对齐参考 `total_batch_idx` vs `batch_idx` 分支）。

### 接线
- `fit_loop.run()`：算有效批数后调 `_setup_val_check_batch`。
- `training_epoch_loop._run_validation` 的 `finally`：时间态验证后重置 `_last_val_time`（重开 wall-clock 窗口）。

## 行为对照表

| val_check_interval | 效果 | 验证 |
|---|---|---|
| `1.0`（默认） | 每 epoch 末验证一次 | ✅ vcb=有效批数 |
| `0.5` | 每 epoch 验证 2 次 | ✅ vcb=int(4*0.5)=2 |
| `2`（int） | 每 2 个 batch 验证 | ✅ |
| `check_val_every_n_epoch=2` | 仅第 2/4/... epoch | ✅ |
| `int > 批数`（有 epoch 门控） | `ValueError` | ✅ |
| `check_val_every_n_epoch=None` + int | 跨 epoch 全局计数触发 | ✅ |
| 时间态 `"00:00:00:00"` | 每 batch（0 秒预算） | ✅ |
| `limit_val_batches=0` | 禁用验证 | ✅ |

## 验证

- **新增** `tests/test_val_check_interval.py`（9 例，覆盖上表全部 + epoch 门控单测）。全过。
- **全量核心回归** `pytest tests/ --ignore=tests/compat` → **154 passed / 24 skipped / 0 failed**（含 F9 的 13+ 例、F3 新增 9 例）。默认行为改变（fit 现在真正跑验证）**未破坏任何既有用例**——它们要么禁用 val、要么无 val_dataloader、要么 validation_step 合法。
- `tests/compat/` 全挂仍为 `msgpack` 缺包（基线同样挂，非回归）。
- `ruff check` + `ruff format` 全过；`grep -rniI lightning ocean/` 无命中。

## 交付状态

- 续在分支 `fix/logger-result-collection`（未 commit，用户自行 commit + 开 PR）。可与 F9 合并为一个 PR，或用户拆分。
- 约定遵守：源码无 Lightning 字样；commit/PR 不加 AI 署名。

## 关联/后续

- 连带修正：`training_epoch_loop._update_lr_schedulers` 的 plateau monitor 读取已在 F9 改为 `callback_metrics`。
- **未做（可作 D2 收尾）**：`_DataConnector.on_trainer_init` 里对 `check_val_every_n_epoch` 必须为 int、以及 `check_val_every_n_epoch=None` 时 float `val_check_interval` 的**构造期**参数校验（参考在此处抛 `MisconfigurationException`）。当前 F3 在运行期（fit 时）对非法 int 抛错，未覆盖构造期早失败。
