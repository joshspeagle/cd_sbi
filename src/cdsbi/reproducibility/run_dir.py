"""Atomic writes and STATUS file management for run directories."""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Union


class RunStatus(str, Enum):
    RUNNING = "RUNNING"
    OK = "OK"
    FAILED = "FAILED"


class RunDir:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def set_status(self, status: RunStatus) -> None:
        self.write_atomic(self.path / "STATUS", status.value + "\n")

    def write_atomic(self, target: Path, content: Union[str, bytes]) -> None:
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        mode = "wb" if isinstance(content, bytes) else "w"
        with open(tmp, mode) as f:
            f.write(content)
        os.replace(tmp, target)  # atomic on POSIX and modern Windows
