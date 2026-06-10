"""Accelerators package for paddleOcean."""

from ocean.accelerators.accelerator import Accelerator
from ocean.accelerators.cpu import CPUAccelerator
from ocean.accelerators.cuda import CUDAAccelerator, GPUAccelerator
from ocean.accelerators.rocm import ROCmAccelerator
from ocean.accelerators.xpu import XPUAccelerator
from ocean.accelerators.ipu import IPUAccelerator
from ocean.accelerators.custom_device import CustomDeviceAccelerator
