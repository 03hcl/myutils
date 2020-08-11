from bisect import bisect_right
from typing import Any, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from ..util_logger import UtilLogger


class LazyLoadedDataset(Dataset):

    def __init__(self, file_path_list: Iterable[str], logger: Optional[UtilLogger] = None):

        super(LazyLoadedDataset, self).__init__()

        # if logger:
        #     logger.info("LazyLoadedDataset を生成します。")

        self.file_path_list: Tuple[str, ...] = tuple(file_path_list)
        self.file_count: int = len(self.file_path_list)

        self.file_index_heads: List[int] = [0] * self.file_count
        self.length: int = 0

        for i in range(self.file_count):
            self.file_index_heads[i] = self.length
            self.length += len(torch.load(self.file_path_list[i]))
            if logger:
                logger.info("[{}: {}] (file_path = {})".format(
                    self.file_index_heads[i], self.length, self.file_path_list[i]))

        self.cache: Optional[Dataset] = None
        self.cache_index: int = -1

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Any:
        file_index: int = bisect_right(self.file_index_heads, index) - 1
        if file_index != self.cache_index:
            self.cache = torch.load(self.file_path_list[file_index])
            self.cache_index = file_index
        return self.cache[index - self.file_index_heads[file_index]]

    def clear_cache(self):
        self.cache = None
        self.cache_index = -1
