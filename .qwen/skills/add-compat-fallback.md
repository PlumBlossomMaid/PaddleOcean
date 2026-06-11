# paddleOcean Skill — 添加新 API Fallback

在 `ocean/_compat/` 下为低版本 Paddle 缺失的 API 添加纯 Python fallback。

## 检测流程

```python
from ocean._compat.version import api_available, version_gte

# 检查 API 是否存在
if api_available("paddle.xxx"):
    # 直接用 paddle 版本
    ...
else:
    # 用纯 Python 实现
    ...
```

## 步骤

1. 在 `ocean/_compat/tensor.py`（或新建文件）中实现 fallback
2. 在 `ocean/_compat/__init__.py` 中导出
3. 在 `ocean/__init__.py` 顶部导入并加入 `__all__`
4. 在 `tests/test_api_coverage.py` 中添加测试用例
5. 运行 `pytest tests/ -v --timeout=120`
6. ruff check && ruff format

## 规范

- fallback 必须只用 Paddle 已有 API 实现，不能引入 numpy 等外部依赖
- 尽量保持与高版本 paddle 行为一致
- 0D Tensor 标量行为可以不同
- C++ CUDA 算子是最后手段，优先 Python
