from contextlib import nullcontext

import torch

try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    torch_npu = None


def get_runtime_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_inference_dtype(device):
    if device != "cuda":
        return torch.float32

    try:
        capability = torch.cuda.get_device_capability()
        return torch.bfloat16 if capability[0] >= 8 else torch.float16
    except Exception:
        # torch_npu may route CUDA APIs to NPU without supporting capability probing.
        return torch.bfloat16


def autocast_context(device, dtype):
    if device != "cuda":
        return nullcontext()
    return torch.cuda.amp.autocast(dtype=dtype)


def seed_runtime(seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def safe_empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()