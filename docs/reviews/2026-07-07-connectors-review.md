# Ocean `trainer/connectors/` 行级 Review 报告

- **日期**：2026-07-07
- **范围**：`ocean/trainer/connectors/__init__.py`（533 行，含 6 个 connector：Data / Logger / Callback / Checkpoint / Signal / Accelerator）
- **对照参考**：`lightning-ref/src/lightning/pytorch/trainer/connectors/`（5 个独立文件 + `logger_connector/` 子目录，含 `result.py` 的 `_ResultCollection`）
- **主分支状态**：`master @ 35a6712`（F1 循环合并 PR #32 已合入）
- **分级**：🔴 严重（静默错误结果 / 数据丢失） / 🟠 高危 / 🟡 中危 / 🟢 低危

## 修复进度（分支 `fix/logger-result-collection`，未 commit）

- ✅ **F9-a ~ F9-e**：已由 stage 分离 `_ResultCollection` 根因重构解决（2026-07-07），见 `2026-07-08` 前一份报告 `2026-07-07-f9-refactor.md`。
- ✅ **C1**（Checkpoint restore 早于 model callback attach）：已修（2026-07-08）——`fit` 里把 `_attach_model_callbacks()` 移到 `restore()` 之前，模型 `configure_callbacks()` 返回的 callback 现在能正确恢复其 checkpoint 状态。回归守卫 `tests/test_checkpoint_callback_restore.py`（旧顺序会失败）。
- ✅ **D1**（prepare_data 无 rank 门控）：已修（2026-07-08）——`_DataConnector.prepare_data` 加 `local_rank`/`node_rank` + `prepare_data_per_node` 门控 + barrier；非准备 rank 在 barrier 等待。连带修 **attach_data 此前根本不调 prepare_data**（fit 路径下载被跳过、setup 早于 prepare），现改为 prepare→setup 顺序。3 处 inline `datamodule.prepare_data()`（validate/test/predict）统一走门控路径。守卫 `tests/test_prepare_data_gating.py`（无门控时非零 rank 会失败）。
- ✅ **D2**（val_check 构造期校验）：已修（2026-07-08）——`_DataConnector.on_trainer_init` 加 3 条构造期校验（`check_val_every_n_epoch` 必须 int；`check_val_every_n_epoch=None` 时 `val_check_interval` 不能是 float；`reload_dataloaders_every_n_epochs` 必须 int≥0），抛 `MisconfigurationException`。把 F3 的运行期报错前移为构造期早失败。守卫 `tests/test_trainer_config_validation.py`。
- ✅ **C2**（dump/restore 存取不对称）：已修（2026-07-08）——`restore` 补齐 hparams、datamodule state、`on_load_checkpoint(ckpt)` 三条恢复分支，与 `dump_checkpoint` 对称。连带修 **latent bug**：`Model` 基类此前无 `on_load_checkpoint`（但 `core/saving.py:50` 已在调它 → `load_from_checkpoint` 必崩），已补基类空实现。守卫 `tests/test_checkpoint_restore_symmetry.py`。
- ✅ **C3**（max_epochs 续训防呆）：已修（2026-07-08）——restore 时若 `ckpt["epoch"] > max_epochs` 抛 `MisconfigurationException`（`==` 允许，边界续训对齐参考）。
- ✅ **CB1/CB2**（callback 去重 / 排序）：已修（2026-07-08）——`_attach_model_callbacks` 改为按类型去重（model callback 覆盖同类型/父类型 trainer callback，不再并存重复 ModelCheckpoint）；新增 `_reorder_callbacks`（tuner 最前、checkpoint 最后），`on_trainer_init` 与 attach 后都应用。守卫 `tests/test_callback_ordering.py`。
- ✅ **S1**（SignalConnector 空壳）：已实现（2026-07-08）——`register_signal_handlers` 装 SIGTERM handler（set `received_sigterm` + `should_stop` → fit 循环在 epoch 边界优雅停止，末轮 checkpoint 仍可跑）；`teardown` 还原原 handler；`_fit_impl` 里在 run 前注册。守卫 `tests/test_signal_connector.py`（含真实 `os.kill(SIGTERM)`）。
- ✅ **A1 / A3**（_set_flags 重复 + torch 措辞）：已修（2026-07-08）——删除 deterministic 分支的重复设置块；docstring 去掉 "Mirrors paddleOcean's _set_torch_flags"。
- 🔄 **未修**：C4（weights_only 语义）。
- ✅ **A2**（fleet init 静默吞异常）：已修（2026-07-08）——`except Exception` 改为 `warnings.warn`，提示 fleet 初始化失败及排查方向。
- ✅ **D3**（attach_data 分支不对称）：已修（2026-07-08）——datamodule 分支也初始化 test/predict dataloader 通道，与显式分支对称。
- 📝 **C4 评估保留**：`restore(ckpt_path)` 以 `None` 调用，`if not weights_only` 将 None 当 False 走全量恢复，行为正确，无需改。
- **connectors 子系统 review + 修复至此基本完成**（🔴🟠🟡 全部处理，🟢 评估保留）。
- 备注：(1) **两套 checkpoint 系统**（`core/saving.py` 用 `hyper_parameters` 键 vs connector 用 `hparams` 键）——F1/F9 同族，建议后续统一。(2) 源码大量 docstring 前缀 `paddleOcean`（如 `ocean/__init__.py` "inspired by paddleOcean"、`gear_wrappers.py` "paddleOcean Fabric"、`core/optimizer.py` "paddleOceanOptimizer"）疑似 Lightning→paddleOcean 的粗暴 find-replace 残留（"Fabric" 是对标框架子项目名），自指且别扭，建议统一改为中性措辞或项目正名。非本轮范围。

## TL;DR

- 本轮起于 loops review 的 **F9 根因**（metrics buffer 未按 stage 分离）。深挖后确认 F9 不是单点 bug，而是与 F1 **同构的"两套实现"问题**：`_LoggerConnector` 自带一套 `_metrics_buffer` + reducer 是**死代码**，真正跑的是 `Trainer._log_metrics_buffer`，且缺一个按 stage 分离的结果收集抽象。
- 结构差异：Ocean 用一个 533 行 `__init__.py` 塞 6 个 connector；参考实现是 5 个独立文件 + `logger_connector/` 子目录，且用 `_ResultCollection`（train/eval 天然分离、batch_size 加权归约、支持 min/max/sum reduce）。
- 近 30 commit 中 `3695a31`（清 val 防泄漏）、`704f444`（sanity step-0 flush）、`bdf7ac4`（从 `_compute_epoch_metrics` 填 logged_metrics）三次都在绕 F9 打补丁——典型的"补丁而非根因"。
- **建议主线**：引入 Ocean 原生的 stage 分离 `_ResultCollection`，作为唯一 metrics owner，删除两套 ad-hoc buffer——一次性化解 F9-a~F9-d。做法对标 F1 的"合并两套实现"。

---

## 🔴 严重（会导致训练结果错误）

### F9-a. mid-epoch validation 会静默清掉本 epoch 早期训练 loss 样本

**位置**：
- `ocean/loops/training_epoch_loop.py:184` — `_run_validation()` 调 `trainer._compute_epoch_metrics()`
- `ocean/trainer/__init__.py:688-694` — `_compute_epoch_metrics()` 末尾 `self._log_metrics_buffer.clear()`

**证据 / 数据流**：训练每 batch 在 `_log_metric`（`trainer/__init__.py:677`）把 `loss` append 进 `_log_metrics_buffer`。当 `val_check_interval` 命中，`_run_validation()`（`training_epoch_loop.py:158`）跑完 val 后调 `_compute_epoch_metrics()`，其最后一行**无差别** `clear()` 整个 buffer——包括 val 触发点**之前**累积的训练 `loss`。

**后果**：epoch 末的训练 loss 只对 val 触发点**之后**的 batch 求均值，前 N 个 batch 的 loss 全丢。开了 mid-epoch val 时，日志里的 epoch train loss 系统性偏差，且随 `val_check_interval` 位置漂移。属**静默正确性 bug**（能跑，指标不对）。

**对照参考实现**：train/eval 指标分别存于各自的 `_ResultCollection`（`result.py`），val 的 reset 只作用于 eval 结果，绝不触碰训练累积。

**建议**（根因方向，见文末统一方案）：train 与 val 的 metric 状态物理隔离；`_compute_epoch_metrics` 只归约当前 stage 的 buffer。

---

### F9-b. 两套 metrics buffer 并存，connector 自带的那套是死代码

**位置**：
- `ocean/trainer/connectors/__init__.py:79` — `_LoggerConnector._metrics_buffer`
- `:130` `update_train_step_metrics`（仅 `pass`）、`:134-141` `update_train_epoch_metrics`（reducer）
- 真正在跑的：`ocean/trainer/__init__.py:261` `Trainer._log_metrics_buffer` + `:688` `_compute_epoch_metrics`

**证据**：`grep -rn "update_train_epoch_metrics\|update_train_step_metrics" ocean/ tests/` → **零外部调用者**（只在 connector 自身定义处出现）。connector 的 `_metrics_buffer` 经 `log_metric_value`（`:149-151`）被**写入**，但从不为其目的被**读取**，只在 `reset_validation_metrics` 里被 `clear`。epoch 归约实际走的是 Trainer 的另一套 buffer。

**后果**：与 F1（`_TrainingEpochLoop` 死代码）**同构**——两套实现只有一套在跑，维护者容易改错那一套；死 reducer 还带着一个看似正确、实则从不生效的 stage 无关均值逻辑，误导阅读。

**建议**：删除 `_LoggerConnector._metrics_buffer` + `update_train_step_metrics` + `update_train_epoch_metrics`，把唯一 metrics owner 收敛到根因方案的 `_ResultCollection`；`log_metric_value` 只保留 callback/logged/pbar 三个即时快照 dict。

---

## 🟠 高危

### F9-c. epoch 均值是无权重 `sum/len`，末批不均时偏差

**位置**：`ocean/trainer/__init__.py:690` — `mean_val = float(sum(values)) / len(values)`

**问题**：`values` 是每个 step 已经算过一次均值的标量，epoch 层再对这些 step 均值等权平均。当各 step 的 batch_size 不等（尤其 `drop_last=False` 的末批、或梯度累积尾批），会对小 batch 过度加权。

**对照参考实现**：`result.py:234,251` 做 batch_size 加权——`value * batch_size` 累加、除以 `cumulated_batch_size`。

**建议**：根因方案的 `_ResultCollection` 内建 batch_size 加权归约；过渡期可在 `_log_metric` 记录每个 value 的 batch_size 一并加权。

---

### F9-d. `reset_validation_metrics` 用整表 `.clear()`，而非按 stage 清

**位置**：`ocean/trainer/connectors/__init__.py:103-112`

**问题**：为阻止 `loss/val`（不以 `val/` 前缀）泄漏进训练 step-0 flush，该方法直接 `.clear()` 整个 `_logged_metrics` + `_metrics_buffer`。这是对"缺 stage 维度"的钝性补偿——注释本身也承认是为处理 `loss/val` 这类命名。commit `3695a31` 即此补丁。

**后果**：清得过狠，任何跨 val 存活的训练侧 logged metric 也被一并抹掉；且依赖调用时机精确（sanity 前后各清一次，见 `trainer/__init__.py:807,836`），脆弱。

**建议**：根因方案下，val reset 只作用于 eval stage 的结果收集，训练侧不受影响，此方法可删。

---

## 🟡 中危

### F9-e. `log_metrics` 的 rank 过滤下沉到各 logger，connector 层无 `should_update_logs` 门控

**位置**：`ocean/trainer/connectors/__init__.py:97-101`

**问题**：注释声明"delegates to each logger (which filter by rank internally)"。参考实现把 `should_update_logs`（按 `log_every_n_steps` 节流）+ rank-zero 门控集中在 connector 层（`logger_connector.py:51-65,100-131`），Ocean 把节流散落在 `training_epoch_loop._maybe_log_metrics:152` 和各 logger 内部，职责分散、易漏。

**建议**：把 `should_update_logs` 节流与 rank-zero 门控收敛到 `_LoggerConnector.log_metrics` 单点。

---

## 其余 5 个 connector

### `_CheckpointConnector`

#### C1. 🟠 `restore()` 在 model callbacks attach 之前跑，模型自定义 callback 的 checkpoint 状态被静默丢弃

**位置**：`ocean/trainer/__init__.py:497`（restore）vs `:500`（`_attach_model_callbacks`）+ `connectors/__init__.py:233-237`

**问题**：fit 流程里先 `restore(ckpt_path)`（497），后 `_attach_model_callbacks()`（500）。而 `restore` 恢复 callback 状态时遍历的是 `self.trainer.callbacks`（`connectors/__init__.py:234`），此刻**还没**包含 `model.configure_callbacks()` 返回的 callback。于是模型侧定义的 callback（如自定义 EMA、自定义 ModelCheckpoint）续训时其 `state_dict` 不会被恢复。

**对照参考实现**：restore 的模块/callback 恢复在 setup（含 callback attach）之后进行（`_restore_modules_and_callbacks` 在 `resume_start` 后调 `restore_callbacks`）。

**建议**：把 `_attach_model_callbacks()` 移到 `restore()` 之前，或让 restore 的 callback 恢复延后到所有 callback attach 完成之后。

#### C2. 🟡 dump 存了 datamodule / hparams，restore 从不恢复（存取不对称）

**位置**：dump `connectors/__init__.py:286-293`（datamodule state、hparams）vs restore `:210-253`（只恢复 state_dict / optimizer / lr_scheduler / callbacks / precision）

**问题**：`dump_checkpoint` 保存了 `datamodule.state_dict()` 和 `model.hparams`，但 `restore()` 没有对应的读取分支。续训时 datamodule 内部状态（如采样进度）与 hparams 丢失。

**建议**：restore 补 datamodule / hparams 恢复分支，或 dump 侧明确不存（保持对称）。

#### C3. 🟡 缺 `max_epochs < current_epoch` 的续训防呆

**位置**：`connectors/__init__.py:245-246`（直接 `current_epoch = ckpt["epoch"]`）

**问题**：从 epoch=50 的 checkpoint 续训但 `Trainer(max_epochs=10)` 时，参考实现会抛 `MisconfigurationException`（`checkpoint_connector.py:356-365`）；Ocean 直接赋值，训练循环随即空转/立即结束，无提示。

**建议**：restore 后校验 `current_epoch <= max_epochs`，否则报错。

#### C4. 🟢 `restore(ckpt_path)` 恒以 `weights_only=None` 调用

**位置**：`trainer/__init__.py:497` → `connectors/__init__.py:210` `if not weights_only`

**说明**：`None` 经 `not None == True` 走全量恢复，行为正确但语义含糊；建议显式传 `weights_only=False` 或从 checkpoint 内容推断。

---

### `_DataConnector`

#### D1. 🟠 `prepare_data()` 无 rank / node 门控，多卡下所有 rank 同时下载 → 竞争/损坏

**位置**：`connectors/__init__.py:40-42`

**问题**：`prepare_data` 直接 `datamodule.prepare_data()`，无 `local_rank==0` / `prepare_data_per_node` 门控，也无分布式 barrier。DDP 下每个 rank 都会执行下载/预处理，共享文件系统上并发写同一路径会竞争或写坏。

**对照参考实现**：`data_connector.py:80-102` 用 `_InfiniteBarrier` + `local_rank_zero` / `global_rank_zero` 精确门控，且同时调 datamodule **和** module 两个 `prepare_data` hook。

**建议**：加 rank-zero 门控 + barrier；补 model 侧 `prepare_data` hook。

#### D2. 🟠 `on_trainer_init` 不校验 `val_check_interval` / `check_val_every_n_epoch` 非法配置（呼应 loops F3）

**位置**：`connectors/__init__.py:30-38`

**问题**：参考实现在此处校验：`check_val_every_n_epoch` 必须是 int；`check_val_every_n_epoch=None` 时 `val_check_interval` 不能是 float（`data_connector.py:60-69`）。Ocean 全盘照收，下游 `_should_check_val_step`（`trainer/__init__.py:781-784`）只认 int，float/时间态被静默吞掉——与 loops review 的 F3 同源。

**建议**：在 `on_trainer_init` 加同样的类型校验并抛错，把 F3 的静默失败前移为显式报错。

#### D3. 🟡 datamodule 分支只 `setup("fit")`，且不设 test/predict dataloaders（分支不对称）

**位置**：`connectors/__init__.py:54-63`

**问题**：datamodule 分支硬编码 `datamodule.setup("fit")`，只填 `train_dataloader` + `val_dataloaders`；非 datamodule 分支（61-63）却填了 test/predict。虽然 `test()`/`validate()` 各自会重新 `setup`（`trainer/__init__.py:538-539,564-568`），但 `attach_data` 两分支语义不一致，易埋坑。

**建议**：按调用 stage 传入 `setup(stage)`，或在 datamodule 分支也惰性准备 test/predict 通道。

---

### `_CallbackConnector`

#### CB1. 🟡 `_attach_model_callbacks` 不去重，configure_callbacks 与默认/用户 callback 可能重复

**位置**：`connectors/__init__.py:189-195`

**问题**：直接 `trainer.callbacks.extend(extra)`，不做同类型去重。若模型 `configure_callbacks()` 返回一个 `ModelCheckpoint`，而 `on_trainer_init`（`:177-178`）已因 `enable_checkpointing` 加了默认 `ModelCheckpoint`，会同时存在两个，保存两份 checkpoint。

**对照参考实现**：模型侧 callback 覆盖 trainer 侧同类型 callback（去重合并）。

**建议**：extend 后按类型去重，模型侧优先。

#### CB2. 🟡 callback 无排序，ModelCheckpoint 未保证最后执行

**位置**：`connectors/__init__.py:168-195`（append 顺序即执行顺序）

**问题**：参考实现会重排 callback，确保 checkpoint 类 callback 在末尾（依赖其它 callback 先更新 metrics）。Ocean 按 append 顺序执行，若 ModelCheckpoint 早于更新 `callback_metrics` 的 callback，`monitor` 读到上一步的旧值。

**建议**：在 attach 完成后按优先级排序（checkpoint / 用户 callback / progress bar 的相对次序对齐参考）。

---

### `_SignalConnector`

#### S1. 🟡 整体是空壳，SIGTERM 优雅退出未实现；`received_sigterm` 恒为 False

**位置**：`connectors/__init__.py:312-323`（`register_signal_handlers` / `teardown` 均 `pass`）

**问题**：`received_sigterm` 在 `__init__` 置 False 后**无处置 True**，任何依赖它做优雅停止/存档的逻辑都是死判断。收到 SIGTERM（如集群抢占、`kill`）时不会保存 checkpoint。参考实现注册 SIGTERM/SIGUSR handler，broadcast 停止信号并存档。

**说明/建议**：Paddle 无 SLURM 场景可简化，但至少应实现 SIGTERM → set flag → 循环边界 checkpoint 保存 + 恢复原 handler 的 teardown。若暂不实现，建议删掉 `received_sigterm` 这个误导性的死字段，或在 docstring 注明未实现。

---

### `_AcceleratorConnector`

#### A1. 🟡 `_set_flags` 的 deterministic 分支重复设置两遍（复制粘贴残留）

**位置**：`connectors/__init__.py:501-514` 与 `522-533`

**问题**：第一块 `if deterministic is True or deterministic == "warn":`（501）已设置 `FLAGS_cudnn_deterministic` + `CUBLAS_WORKSPACE_CONFIG`；第二块 `if deterministic is True:`（522）/ `elif deterministic == "warn":`（528）**又设一遍**完全相同的 flag。属合并/复制残留，无功能收益，徒增困惑。

**建议**：删除 522-533 冗余块，保留 501-514（其含 benchmark 默认值与冲突告警逻辑）。

#### A2. 🟢 `strategy="fleet"` 分支 fleet 初始化失败被静默吞掉

**位置**：`connectors/__init__.py:452-462`

**问题**：`odist.fleet.init(is_collective=True)` 包在 `try/except Exception: pass` 里，失败后仍返回 DDPStrategy，后续通信会以更隐晦的方式崩。

**建议**：至少 warn，或让 fleet init 失败显式抛错。

#### A3. 🟢 docstring 残留框架专有措辞

**位置**：`connectors/__init__.py:496` docstring "Mirrors paddleOcean's `_set_torch_flags`"

**说明**：措辞引用了外部框架的内部函数名。按项目约定（源码不出现对标框架专有名），建议改为中性描述，如"设置确定性/基准模式 flag"。（注：`grep -rniI lightning ocean/` 当前无命中，本条属 torch 措辞的收尾。）



## 根因方案：引入 stage 分离的 `_ResultCollection`（对标 F1 的"合并两套实现"）

**目标**：一次性化解 F9-a ~ F9-d，消除 F9-b 的死代码。

**要点**：
1. 新增 Ocean 原生 `_ResultCollection`（放到 `ocean/trainer/connectors/logger_connector/result.py`，把 6-in-1 巨文件按参考结构拆分为子目录）：
   - 按 `(stage, fx, name)` 键存储，`training=True/False` 天然分离，val/test 归约绝不触碰训练累积（解 F9-a、F9-d）。
   - 内建 batch_size 加权 mean + min/max/sum reduce_fx（解 F9-c）。
   - on_step / on_epoch 分离缓存。
2. 让 `_ResultCollection` 成为**唯一** metrics owner：删除 `Trainer._log_metrics_buffer`、`_LoggerConnector._metrics_buffer` 及其死 reducer（解 F9-b）。
3. `_LoggerConnector.log_metrics` 收敛节流 + rank 门控（解 F9-e）。
4. 因地制宜：Paddle 无 `torchmetrics.Metric` 基类，`paddlemetrics` 分支（`trainer/__init__.py:643-651`）仍走 `.compute()` 委托，接入新 `_ResultCollection` 的 metric-object 通道。

**落地节奏**（沿用 F1 流程）：报告先行 → 用户确认 → 单独开分支实现 + 新增测试（覆盖 mid-epoch val 不清训练 buffer、batch_size 加权均值、val 指标不泄漏）→ 全量回归 → 用户自行开 PR。

**约定**：源码注释/变量/docstring 不出现 Lightning 字样（改称"参考实现"）；commit / PR 不加任何 AI 工具署名。当前 `ocean/` 源码已无 Lightning 残留（`grep -rniI lightning ocean/` 无命中，`d3d94a1` 已清理），需保持。

---

## 近 30 commit 复核（判断补丁 vs 根因）

围绕 metrics/logger 的补丁式提交：

| commit | 说明 | 定性 |
|--------|------|------|
| `3695a31` | clear val/test metrics after validation to prevent leaking | F9-d 的钝性补丁（绕根因） |
| `704f444` | flush sanity check metrics to logger at step 0 | 绕 sanity/train buffer 混用 |
| `bdf7ac4` | populate logged_metrics from `_compute_epoch_metrics` | 补 logged_metrics 未填（两套 buffer 副作用） |
| `5746017` | respect logger=False in `_log_metric` | 局部修正 |

结论：三处补丁都在同一处缺失抽象（stage 分离的结果收集）周围打转，与 F1 前的形态一致。建议按根因方案合并，后续补丁自然消失。
