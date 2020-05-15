from dataclasses import dataclass
from typing import Optional

from ..trainer_base import DataTensorLike, ScoreLike


@dataclass
class TrainResult:
    output: DataTensorLike
    loss: ScoreLike
    target: Optional[DataTensorLike]
