import torch

from contextlib import nullcontext
from typing import Any, Literal


RuntimeMode = Literal['training', 'inference']
TorchCompileMode = Literal['default', 'reduce-overhead', 'max-autotune']


_BF16_ENABLED = False


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
    )  # type: ignore[return-value]


def set_bf16_enabled(enabled: bool = True) -> None:
    """Enable or disable BF16 autocast for CUDA execution.

    Args:
        enabled: Whether BF16 autocast should be enabled.
    """
    global _BF16_ENABLED
    _BF16_ENABLED = enabled


def is_bf16_enabled() -> bool:
    """Return whether BF16 autocast is currently enabled."""
    return _BF16_ENABLED


def autocast_context(device: torch.device) -> Any:
    """Return the active autocast context for the given device.

    BF16 autocast is only enabled on CUDA devices when ``set_bf16_enabled`` has
    been turned on. Otherwise, this returns a no-op context manager.

    Args:
        device: Device on which the upcoming operations will run.

    Returns:
        A context manager that enables BF16 autocast when configured.
    """
    if _BF16_ENABLED and device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return nullcontext()
