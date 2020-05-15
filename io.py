from pathlib import Path
from typing import Optional


def get_common_root_directory(path_str: str) -> Path:
    path: Optional[Path] = None
    for p1, p2 in zip(Path(__file__).resolve().parts, Path(path_str).resolve().parts):
        if p1 == p2:
            path = path.joinpath(p1) if path else Path(p1)
        else:
            return path
