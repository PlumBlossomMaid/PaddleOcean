"""Strategies package for paddleOcean."""

from ocean.strategies.strategy import Strategy
from ocean.strategies.single_device import SingleDeviceStrategy
from ocean.strategies.parallel import ParallelStrategy
from ocean.strategies.ddp import DDPStrategy
from ocean.strategies.deepspeed import DeepSpeedStrategy
from ocean.strategies.fsdp import FSDPStrategy
