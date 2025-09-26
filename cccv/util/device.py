import sys
import warnings

import torch


def default_device() -> torch.device:
    if sys.platform != "darwin":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        try:
            return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        except Exception as e:
            warnings.warn(f"[CCCV] {e}, MPS is not available, use CPU instead.", stacklevel=2)
            return torch.device("cpu")


DEFAULT_DEVICE = default_device()
