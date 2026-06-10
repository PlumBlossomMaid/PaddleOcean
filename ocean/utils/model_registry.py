"""Model registry for paddleOcean.

Stores and retrieves model configurations.
"""

from typing import Any, Optional


class ModelRegistry:
    """Global registry for model classes and configurations.

    Usage::
        @ModelRegistry.register("my_model")
        class MyModel(ocean.Model): ...

        model = ModelRegistry.get("my_model")()
    """

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: Optional[str] = None) -> Any:
        """Decorator to register a model class.

        Args:
            name: Name to register under (defaults to class name).
        """
        def decorator(model_cls: type) -> type:
            key = name or model_cls.__name__
            cls._registry[key] = model_cls
            return model_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """Get a registered model class by name.

        Args:
            name: Registered name.

        Returns:
            The model class.

        Raises:
            KeyError: If name is not registered.
        """
        if name not in cls._registry:
            raise KeyError(f"Model '{name}' not registered. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def available(cls) -> list[str]:
        """List all registered model names."""
        return list(cls._registry.keys())
