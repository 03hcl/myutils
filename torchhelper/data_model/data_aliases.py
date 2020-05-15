from typing import Dict, Iterable, Union

import torch
from torch.utils.data import DataLoader, Dataset


DataTensorLike = Union[torch.Tensor, Iterable[torch.Tensor], Dict[str, torch.Tensor]]
DatasetLike = Union[Dataset, Iterable[Dataset], Dict[str, Dataset]]
DataLoaderLike = Union[DataLoader, Iterable[DataLoader], Dict[str, DataLoader]]
NumericScore = Union[int, float, bool]
Score = Union[NumericScore, torch.Tensor]
ScoreDict = Union[Dict[str, int], Dict[str, float], Dict[str, bool]]
ScoreIterable = Union[Iterable[int], Iterable[float], Iterable[bool]]
ScoreLike = Union[NumericScore, ScoreIterable, ScoreDict, DataTensorLike]
