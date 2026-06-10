# paddleOcean Skill — 本地 CI 验证

在提交 PR 前快速验证所有检查项。

## 使用方式

```
/skill local-ci
```

## 执行步骤

```bash
# 1. 代码风格
ruff check ocean/ tests/
ruff format ocean/ tests/ --check

# 2. 类型检查（如果有 mypy）
# mypy ocean/

# 3. 单元测试
python -m pytest tests/ -v --timeout=120

# 4. 导入完整性
python -c "import ocean; print(f'{len(ocean.__all__)} exports OK')"
python -c "import ocean.distributed; print('distributed OK')"

# 5. 分布式模块
python -c "
from ocean.strategies import Strategy, SingleDeviceStrategy, DDPStrategy
from ocean.strategies.deepspeed import DeepSpeedStrategy
from ocean.strategies.fsdp import FSDPStrategy
print('strategies OK')
"
```

## 预期结果

- ruff: 0 errors, 0 warnings
- pytest: 所有测试通过
- 导入: 无 ModuleNotFoundError
