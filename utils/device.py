import torch


def get_device() -> str:
    """Detect and return the best available compute device."""
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print("USING GPU (CUDA)")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("USING MPS (Apple Silicon)")
        return "mps"
    else:
        print("USING CPU")
        return "cpu"
