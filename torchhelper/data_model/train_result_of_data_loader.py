from dataclasses import dataclass
from typing import Optional

from .data_aliases import NumericScore, Score


@dataclass
class TrainResultOfDataLoader:
    data_count: int
    loss: NumericScore
    score: Optional[Score]
