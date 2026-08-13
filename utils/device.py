import torch
import warnings

_BACKEND = None


def _detect_backend():
    """Auto-detect the best available PyTorch backend."""
    try:
        import torch_npu
        if hasattr(torch.npu, 'is_available') and torch.npu.is_available():
            return 'npu'
    except Exception:
        pass

    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def initialize_device(prefer=None):
    """Return a torch.device for the selected backend.

    Args:
        prefer: Optional backend name ('npu', 'cuda', 'mps', 'cpu'). If None,
                auto-detect the best available backend.

    Returns:
        torch.device
    """
    global _BACKEND
    backend = prefer or _BACKEND or _detect_backend()
    _BACKEND = backend

    # Importing torch_npu registers the 'npu' device type with PyTorch.
    # This must happen before torch.device('npu') is called.
    if backend.startswith('npu'):
        try:
            import torch_npu  # noqa: F401
        except Exception:
            pass

    return torch.device(backend)


def current_device_type():
    """Return the type of the currently selected device ('npu', 'cuda', etc.)."""
    return initialize_device().type


def is_npu_available():
    """Return True if an Ascend NPU is available."""
    try:
        import torch_npu
        return hasattr(torch.npu, 'is_available') and torch.npu.is_available()
    except Exception:
        return False