from typing import Any, Callable, Dict, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from .data_model import DataLoaderLike


def create_data_loader_dict(data_loader_like: DataLoaderLike, key_str_func: Callable[[str], str]) \
        -> Dict[str, DataLoader]:
    if isinstance(data_loader_like, DataLoader):
        return {key_str_func(""): data_loader_like}
    if isinstance(data_loader_like, Sequence):
        return {key_str_func(str(k)): v for k, v in enumerate(data_loader_like)}
    if isinstance(data_loader_like, Mapping):
        return {key_str_func(k): v for k, v in data_loader_like.items()}
    raise TypeError


def create_key_str(base: str) -> Callable[[str], str]:
    return lambda key: base + ("_{}".format(key) if key else "")
    # return lambda key: base + (" ({})".format(key) if key else "")


def get_data_length(data: Any) -> int:
    if isinstance(data, torch.Tensor):
        return data.shape[0]
    if isinstance(data, Sequence):
        return data[0].shape[0]
    if isinstance(data, Mapping):
        return len(next(iter(data.values())))
    raise TypeError
