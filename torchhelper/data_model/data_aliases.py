from typing import Dict, Iterable, Union

import torch
from torch.utils.data import DataLoader, Dataset


DataTensorLike: type = Union[torch.Tensor, Iterable[torch.Tensor], Dict[str, torch.Tensor]]
DatasetLike: type = Union[Dataset, Iterable[Dataset], Dict[str, Dataset]]
DataLoaderLike: type = Union[DataLoader, Iterable[DataLoader], Dict[str, DataLoader]]
NumericScore: type = Union[int, float, bool]
Score: type = Union[NumericScore, torch.Tensor]
ScoreDict: type = Union[Dict[str, int], Dict[str, float], Dict[str, bool]]
ScoreIterable: type = Union[Iterable[int], Iterable[float], Iterable[bool]]
ScoreLike: type = Union[NumericScore, ScoreIterable, ScoreDict, DataTensorLike]

TRAIN_KEY_STR: str = "train"
VALIDATION_KEY_STR: str = "val"
TEST_KEY_STR: str = "test"
