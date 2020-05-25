from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class TrainLog:
    data_keys: Tuple[str, ...]
    loss_array: np.ndarray
    score_array: np.ndarray
    score: float
