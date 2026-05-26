"""Device resolution: GPU by default with CPU fallback."""
import torch


def get_device(spec: str = "auto") -> torch.device:
    """Resolve device specification to a torch.device.

    Args:
        spec: Device specification. Options:
            - "auto" (default): CUDA if available, else CPU.
            - "cpu": CPU device (always available).
            - "cuda": CUDA device (raises RuntimeError if not available).

    Returns:
        torch.device: Resolved device.

    Raises:
        ValueError: If spec is not recognized.
        RuntimeError: If spec="cuda" but CUDA is not available.
    """
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec in {"cpu", "cuda"}:
        if spec == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA not available")
        return torch.device(spec)
    raise ValueError(f"Unknown device spec: {spec!r}")
