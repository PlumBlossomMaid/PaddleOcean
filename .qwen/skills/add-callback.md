# paddleOcean Skill — 添加新 Callback

在 `ocean/callbacks/` 下按照 Lightning 风格添加新的回调实现。

## 使用方式

```
/skill add-callback <CallbackName> <钩子列表>
```

## 步骤

1. 读取目标回调的 Lightning 源码（如果有的话）
2. 在 `ocean/callbacks/<name>.py` 中创建新类
3. 更新 `ocean/callbacks/__init__.py` 导出新类
4. 为回调编写 pytest 测试
5. 运行 `pytest tests/ -v --timeout=120` 验证

## 回调规范

- 继承 `Callback` 基类
- 所有参数显式声明，无 `**kwargs`
- 钩子签名：`(self, trainer, model, *args, **kwargs)`
- 生命周期方法全部在 `ocean/callbacks/callback.py` 中定义
- 保持幂等：多次调用 `setup`/`teardown` 不应产生副作用
