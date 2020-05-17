from typing import Any, Callable, Dict

import torch
from torch.utils.data import DataLoader

from .data_model import DataLoaderLike


def create_data_loader_dict(data_loader_like: DataLoaderLike, key_str_func: Callable[[str], str]) \
        -> Dict[str, DataLoader]:
    data_loader_dict: Dict[str, DataLoader] = dict()
    if issubclass(type(data_loader_like), DataLoader):
        data_loader_dict[key_str_func("")] = data_loader_like
    elif issubclass(type(data_loader_like), tuple) or issubclass(type(data_loader_like), list) \
            or issubclass(type(data_loader_like), set):
        for key, value in enumerate(data_loader_like):
            data_loader_dict[key_str_func(str(key))] = value
    elif issubclass(type(data_loader_like), dict):
        for key, value in data_loader_like.items():
            data_loader_dict[key_str_func(key)] = value
    else:
        raise TypeError
    return data_loader_dict


def create_key_str(base: str) -> Callable[[str], str]:
    return lambda key: base + (" ({})".format(key) if key else "")


def get_data_length(data: Any) -> int:

    if issubclass(type(data), dict):
        return len(next(iter(data.values())))

    # noinspection PyUnusedLocal
    tensor: torch.Tensor

    if issubclass(type(data), torch.Tensor):
        tensor = data
    elif issubclass(type(data), tuple) or issubclass(type(data), list) or issubclass(type(data), set):
        tensor = data[0]
    else:
        raise TypeError

    return tensor.shape[0]
