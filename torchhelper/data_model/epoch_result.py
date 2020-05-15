from dataclasses import dataclass
from typing import Optional

from torch.utils.data import DataLoader

from .data_aliases import NumericScore


@dataclass
class EpochResult:
    data_loader: DataLoader
    data_count: int
    loss: NumericScore
    score: Optional[NumericScore]
