# paddleOcean Skill — XPU P800 适配指南

昆仑芯 P800 在 Ocean 中的完整适配流程，涵盖环境配置、加速器实现、测试跳过和已知问题。

## 参考文档

- [PaddlePaddle XPU P800 官方教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/hardware_support/xpu/xpu-p800_paddle_tutorial_cn.html)

## 关键事实

| 项目 | 说明 |
|------|------|
| 设备名 | `"xpu"`（硬编码在 Paddle 主框架） |
| 检测 API | `paddle.is_compiled_with_xpu()` |
| Place 类型 | `paddle.XPUPlace(0)` |
| set_device | `paddle.set_device("xpu:0")` |
| CustomDevice? | **否** — 不在 `paddle.device.get_all_custom_device_type()` 中 |
| 安装包 | `paddlepaddle-xpu`（非 `paddlepaddle-gpu`） |
| 监控工具 | `xpu_smi` |

## 安装步骤

```bash
# 安装 XPU 版 Paddle（nightly）
python -m pip install --pre paddlepaddle-xpu \
  -i https://www.paddlepaddle.org.cn/packages/nightly/xpu-p800/

# 安装 Ocean（开发模式）
pip install -e . --no-build-isolation
```

## 环境变量

以下是 P800 必需的运行时环境变量（必须在 Python 进程启动前设置）：

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

## 加速器实现要点

`ocean/accelerators/xpu.py` 中的 `XPUAccelerator`：

```python
class XPUAccelerator(Accelerator):
    def setup_device(self, device=None):
        return paddle.XPUPlace(0)

    def setup(self, trainer):
        if paddle.is_compiled_with_xpu():
            paddle.device.set_device("xpu:0")

    @staticmethod
    def is_available():
        return paddle.is_compiled_with_xpu()

    @staticmethod
    def auto_device_count():
        return 1 if paddle.is_compiled_with_xpu() else 0
```

**关键区别**：XPU 使用 `paddle.is_compiled_with_xpu()` 而非 `paddle.device.get_all_custom_device_type()`。这是因为它硬编码在 Paddle 主框架中。

## 设备检测链

在 `_resolve_accelerator("auto")` 中，XPU 处于第三优先级：

```
CUDA 可用? → CUDAAccelerator
ROCm 可用? → ROCmAccelerator
XPU 可用?  → XPUAccelerator       ← 昆仑芯
Custom?    → CustomDeviceAccelerator
否则       → CPUAccelerator
```

显式指定 `accelerator="xpu"` 可跳过检测链。

## 测试注意事项

### skip_on_custom_device 对 XPU 无效

`skip_on_custom_device()` 检查 `paddle.device.get_all_custom_device_type()`，该 API **不包含** `"xpu"`。因此该装饰器不会在 XPU 上触发跳过。

如需在 XPU 上跳过测试，使用：

```python
@RunIf(skip_if=paddle.is_compiled_with_xpu(), reason="Feature not supported on XPU")
def test_something(): ...
```

### float64 支持

`_UNSUPPORTED_FLOAT64_DEVICES` 目前仅包含 `"iluvatar_gpu"`。XPU P800 的 float64 支持程度待实际验证：
- 如果 XPU 缺少 float64 kernel → 需要添加到 `_UNSUPPORTED_FLOAT64_DEVICES`
- 如果 XPU 完整支持 float64 → 无需操作
- 验证方法：运行 `tests/compat/test_window.py` 等依赖 float64 的测试

### Audio compat 测试

`tests/compat/conftest.py` 中的 `in_device_blacklist()` 检查 `paddle.get_device()`。在 XPU 上 `paddle.get_device()` 返回 `"xpu:0"`，该函数检查 `"cpu" not in ...` → 会正确跳过。

## 代码适配步骤总结

1. **安装 paddlepaddle-xpu**（非 paddlepaddle-gpu）
2. **设置环境变量**（见上方列表）
3. **代码无需改动** — CPU 代码几乎直接运行在 XPU 上，唯一差异是 `paddle.set_device("xpu")`
4. **指定 accelerator**：`Trainer(accelerator="xpu")` 或 `Trainer(accelerator="auto")`
5. **运行测试**：`pytest tests/ -v --timeout=120`

## 已知待确认项

- [ ] float64 kernel 支持程度
- [ ] 多卡训练（XPU P800 多卡）是否可用
- [ ] `get_device_stats()` 内存统计 API 是否存在
- [ ] `--skip-double` 类似编译标志是否存在
