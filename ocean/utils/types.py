"""Type definitions for paddleOcean."""

from typing import Any, Mapping, Optional, Union

import paddle

# Basic types
STEP_OUTPUT = Optional[Union[paddle.Tensor, Mapping[str, Any]]]
EVALUATE_OUTPUT = list[Mapping[str, float]]
PREDICT_OUTPUT = Union[list[Any], list[list[Any]]]
TRAIN_DATALOADERS = Any
EVAL_DATALOADERS = Any
