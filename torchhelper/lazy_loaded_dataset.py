from bisect import bisect_right
from typing import Iterable, Optional, Tuple

import torch
from torch.utils.data import TensorDataset

from ..util_logger import UtilLogger, BlankUtilLogger


class LazyLoadedSequentialTensorDataset(TensorDataset):

    def __init__(self, file_path_list: Iterable[str]):

        super(LazyLoadedSequentialTensorDataset, self).__init__()

        self.file_path_list: Tuple[str, ...] = tuple(file_path_list)
        self.file_index_ends: list = []
        self.length: int = 0

        for file_path in self.file_path_list:
            self.file_path_list += self.length
            self.length += len(torch.load(file_path))

        self.cache: Optional[TensorDataset] = None
        self.cache_index: int = -1
        self.clear_cache()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        file_index: int = bisect_right(self.file_index_ends, index) - 1
        if file_index != self.cache_index:
            self.cache = torch.load(self.file_path_list[file_index])
        return self.cache[index - self.file_index_ends[file_index]]

    def clear_cache(self):
        self.cache = None
        self.cache_index = -1
