import torch, psutil, os, signal, gc
from transformers.utils import (
    is_torch_cuda_available
)

def abort_process(pid: int) -> None:
    r"""
    Aborts the processes recursively in a bottom-up way.
    """
    try:
        children = psutil.Process(pid).children()
        if children:
            for child in children:
                abort_process(child.pid)

        os.kill(pid, signal.SIGABRT)
    except Exception:
        pass

def torch_gc() -> None:
    r"""
    Collects GPU or NPU memory.
    """
    gc.collect()
    if is_torch_cuda_available():
        torch.cuda.empty_cache()