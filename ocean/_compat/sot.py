"""Monkey-patch Paddle SOT to fix ``KeyError: 'self'`` when a compiled method
calls another compiled method via ``self.xxx()``.

Root cause
----------
When a framework (e.g. ``ocean.Model.compile()``) replaces instance methods
with ``paddle.jit.to_static`` wrappers::

    self.forward = paddle.jit.to_static(self.forward)
    self.training_step = paddle.jit.to_static(self.training_step)

Accessing ``self.forward`` returns a ``StaticFunction`` that is **bound** to
the instance (via its ``__get__`` descriptor).  However, SOT's
``VariableFactory.from_value`` creates a plain ``UserDefinedFunctionVariable``
(wrapping ``dygraph_function``) for any ``StaticFunction``, regardless of
binding status.  The resulting variable is **not** a ``MethodVariable``, so
``load_method`` treats it as an unbound function and the ``self`` argument is
lost, leading to ``KeyError: 'self'`` inside the inlined callee.

Fix
---
Replace the ``UserDefinedFunctionVariable`` handler in SOT's ``VariableFactory``
dispatch table with a wrapper that detects bound ``StaticFunction`` objects and
returns a proper ``MethodVariable`` instead.

References
----------
- Paddle Issue: https://github.com/PaddlePaddle/Paddle/issues/79325
- Paddle PR: https://github.com/PaddlePaddle/Paddle/pull/79326
"""

import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Sentinel to guard against double-application
_PATCHED = False


def patch_sot():
    """Apply the ``KeyError: 'self'`` monkey-patch at runtime.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _PATCHED
    if _PATCHED:
        return

    try:
        from paddle.jit.dy2static.program_translator import StaticFunction
        from paddle.jit.sot.opcode_translator.executor.tracker import (
            DanglingTracker,
            GetAttrTracker,
        )
        from paddle.jit.sot.opcode_translator.executor.variables.base import (
            VariableFactory,
        )
        from paddle.jit.sot.opcode_translator.executor.variables.callable import (
            MethodVariable,
            UserDefinedFunctionVariable,
        )
    except ImportError:
        logger.debug("SOT monkey-patch skipped: Paddle SOT module not available")
        return

    # Get the original handler from the dispatch table
    original_handler = VariableFactory.mapping_str_func.get("UserDefinedFunctionVariable")
    if original_handler is None:
        logger.debug("SOT monkey-patch skipped: handler not found in dispatch table")
        return

    def _sot_bound_static_fn_handler(value: Any, graph, tracker) -> object | None:
        """Handle ``StaticFunction`` with bound instance.

        Returns a ``MethodVariable`` if the value is a ``StaticFunction``
        bound to an instance, or ``None`` to fall through to the original
        handler.
        """
        if not isinstance(value, StaticFunction):
            return None
        if value.class_instance is None:
            return None

        # Extract the unbound function
        fn = value.dygraph_function
        if inspect.ismethod(fn):
            fn = fn.__func__

        fn_var = UserDefinedFunctionVariable(fn, graph, DanglingTracker())
        instance_var = VariableFactory.from_value(value.class_instance, graph, DanglingTracker())
        method_var = MethodVariable(instance_var, fn_var, graph=graph, tracker=tracker)
        # Real StaticFunction attributes (class_instance, dygraph_function).
        # Guards never evaluated (method_var has non-traceable DanglingTracker).
        instance_var.tracker = GetAttrTracker(method_var, "class_instance")
        fn_var.tracker = GetAttrTracker(method_var, "dygraph_function")
        return method_var

    # Wrap the original handler: try our fix first, then fall through
    def _patched_handler(value, graph, tracker):
        result = _sot_bound_static_fn_handler(value, graph, tracker)
        if result is not None:
            return result
        return original_handler(value, graph, tracker)

    VariableFactory.mapping_str_func["UserDefinedFunctionVariable"] = _patched_handler

    _PATCHED = True
    logger.debug("SOT KeyError 'self' monkey-patch applied")
