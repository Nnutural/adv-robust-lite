from __future__ import annotations

import sys
from pathlib import Path


def add_project_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def add_src_to_path() -> Path:
    root = add_project_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root
