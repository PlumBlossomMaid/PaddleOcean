# paddleOcean Skill — 本地 CI 验证

在提交 PR 前快速验证所有检查项。

## 使用方式

```
/skill local-ci
```

## 执行步骤

```bash
# 1. 代码风格（Paddle 主框架标准：ruff 0.15.0）
ruff check .
ruff format .

# 2. 单元测试
python -m pytest tests/ -v --timeout=120

# 3. 导入完整性
python -c "import ocean; print(f'All {len(ocean.__all__)} exports OK')"
python -c "import ocean; print('randn:', ocean.randn([2,3]).shape); print('nn.Linear:', ocean.nn.Linear(10,2)); print('distributed:', hasattr(ocean.distributed, 'all_reduce'))"

# 4. GPU 验证（如可用）
python -c "import ocean; print('device:', ocean.device.get_device()); print('cuda:', ocean.CUDAAccelerator.is_available())"

# 5. 体验 demo
python E:\\code\\paddle模型大迁移\\体验\\ocean_demo.py --epochs 2
```

## 预期结果

- ruff: 0 errors
- pytest: 57/57 passed
- 导入: 无 ModuleNotFoundError
- GPU: 显示 gpu:0（如可用）
