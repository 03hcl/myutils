from typing import Mapping, Sequence, Union

import torch
from torch.utils.data import DataLoader, Dataset    # , TensorDataset


DataTensorLike: type = Union[torch.Tensor, Sequence[torch.Tensor], Mapping[str, torch.Tensor]]
DatasetLike: type = Union[Dataset, Sequence[Dataset], Mapping[str, Dataset]]
DataLoaderLike: type = Union[DataLoader, Sequence[DataLoader], Mapping[str, DataLoader]]
NumericScore: type = Union[int, float, bool]
Score: type = Union[NumericScore, torch.Tensor]
ScoreMapping: type = Union[Mapping[str, int], Mapping[str, float], Mapping[str, bool]]
ScoreSequence: type = Union[Sequence[int], Sequence[float], Sequence[bool]]
ScoreLike: type = Union[NumericScore, ScoreSequence, ScoreMapping, DataTensorLike]

# TensorDatasetLike: type = Union[TensorDataset, Sequence[TensorDataset], Mapping[str, TensorDataset]]

TRAIN_KEY_STR: str = "train"
VALIDATION_KEY_STR: str = "val"
TEST_KEY_STR: str = "test"
