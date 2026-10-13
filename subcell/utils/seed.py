"""Seeding of Python, NumPy and PyTorch random number generators."""

import os
import random

import numpy as np
import torch


def set_random_seed(seed, deterministic: bool = False) -> None:
    """Seed every generator used by a run.

    With ``deterministic``, cuDNN is also restricted to deterministic kernels and
    float32 matrix multiplication may use TensorFloat-32 ("high" precision).
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("high")
