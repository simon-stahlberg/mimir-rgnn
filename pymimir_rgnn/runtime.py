import torch

from typing import Literal


RuntimeMode = Literal['training', 'inference']
TorchCompileMode = Literal['default', 'reduce-overhead', 'max-autotune']


def set_tf32_enabled(enabled: bool = True) -> None:
    """Enable or disable TF32 globally for float32 CUDA math.

    This affects PyTorch's CUDA matmul and cuDNN execution settings.

    Args:
        enabled: Whether TF32 execution should be enabled.
    """
    torch.set_float32_matmul_precision('high' if enabled else 'highest')
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def is_tf32_enabled() -> bool:
    """Return whether TF32 execution is currently enabled."""
    return (
        torch.backends.cuda.matmul.allow_tf32
        and torch.backends.cudnn.allow_tf32
        and torch.get_float32_matmul_precision() in ('high', 'medium')
    ) # type: ignore
