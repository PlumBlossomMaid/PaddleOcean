"""DeepSpeed equivalent using PaddlePaddle Fleet API for large model training.

PaddlePaddle does not have DeepSpeed. Instead, it provides:
- `paddle.distributed.fleet` for large-scale distributed training
- `paddle.distributed.sharding.group_sharded_parallel` for ZeRO-style sharding

This strategy bridges the gap by providing a DeepSpeed-like API
that maps to Paddle's native distributed capabilities.
"""

from typing import Any, Optional

import paddle

from ocean.strategies.ddp import DDPStrategy
from ocean.utils import MisconfigurationException
from ocean.utils.rank_zero import rank_zero_warn


class DeepSpeedStrategy(DDPStrategy):
    """Large model training strategy using PaddlePaddle Fleet API.

    Maps to Paddle's group_sharded_parallel for ZeRO-stage optimization.

    Args:
        zero_stage: ZeRO optimization stage (1=optimizer, 2=gradient, 3=parameter).
        sharding_level: Paddle sharding level ('os_g', 'os_g2', 'p_g', 'p_g2').
        offload: Enable CPU offloading (requires Paddle >= 3.0).
        **kwargs: Additional DDPStrategy arguments.
    """

    #: Paddle accepts exactly these three sharding levels.
    VALID_LEVELS = ("os", "os_g", "p_g_os")

    ZERO_MAP = {
        1: "os",  # optimizer state sharding
        2: "os_g",  # optimizer + gradient sharding
        3: "p_g_os",  # parameter + gradient + optimizer (full model sharding)
    }

    def __init__(
        self,
        zero_stage: int = 1,
        sharding_level: Optional[str] = None,
        offload: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.zero_stage = zero_stage
        if zero_stage not in self.ZERO_MAP:
            raise MisconfigurationException(f"`zero_stage` must be one of {sorted(self.ZERO_MAP)}, got {zero_stage!r}.")
        self._sharding_level = sharding_level or self.ZERO_MAP[zero_stage]
        if self._sharding_level not in self.VALID_LEVELS:
            raise MisconfigurationException(
                f"`sharding_level` must be one of {list(self.VALID_LEVELS)}, got {self._sharding_level!r}. "
                "These are the levels PaddlePaddle's group_sharded_parallel accepts."
            )
        self._offload = offload
        self._fleet_initialized = False

    def setup(self, trainer: Any) -> None:
        """Setup with Fleet + group sharding."""
        self._init_fleet()
        super().setup(trainer)

    def _init_fleet(self) -> None:
        """Initialize Fleet distributed environment."""
        try:
            import ocean.distributed as odist

            odist.fleet.init(is_collective=True)
            self._fleet_initialized = True
        except Exception as exception:
            # Reported rather than swallowed: without Fleet the run continues
            # unsharded, which looks identical to a working setup until memory
            # or throughput says otherwise.
            rank_zero_warn(f"Fleet initialization failed, training will not be sharded: {exception!r}")

    def _setup_model(self, model: paddle.nn.Layer) -> paddle.nn.Layer:
        """Wrap model with group sharding for ZeRO optimization."""
        if not self._fleet_initialized:
            return model
        try:
            model, optimizer, _ = paddle.distributed.sharding.group_sharded_parallel(
                model,
                self._optimizers[0] if self._optimizers else None,
                level=self._sharding_level,
                offload=self._offload,
            )
            return model
        except Exception as exception:
            rank_zero_warn(f"Group sharding failed, the model is NOT sharded: {exception!r}")
            return model

    def save_checkpoint(self, checkpoint: dict, filepath: str) -> None:
        """Save checkpoint, using group_sharded save if needed."""
        if self.zero_stage >= 2:
            try:
                paddle.distributed.sharding.save_group_sharded_model(
                    self._model,
                    filepath,
                )
                return
            except Exception:
                pass
        super().save_checkpoint(checkpoint, filepath)
