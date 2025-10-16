# src/utils/__init__.py

# This file initializes the utils module.

from .gpu_utils import get_device, to_gpu, from_gpu, gpu_available
from .matrix_ops import keep_S_in, leave_S_out, kl_divergence, marginalize