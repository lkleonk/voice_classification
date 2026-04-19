import gc
import sys

import torch

#################################################################
# GPU handling + user interaction
#################################################################

def set_cuda_to_gpu_nr():
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu') # logic taken from "get_embeddings.py"
    return device


def cleanup_gpu_memory() -> None:
    """Release Python and PyTorch CUDA caches when available."""
    try:
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"GPU cleanup warning: {e}", file=sys.stderr)
