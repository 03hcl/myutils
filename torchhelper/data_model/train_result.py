from dataclasses import dataclass
from typing import Optional

from .data_aliases import DataTensorLike, ScoreLike


@dataclass
class TrainResult:
    output: DataTensorLike
    loss: ScoreLike
    target: Optional[DataTensorLike]
