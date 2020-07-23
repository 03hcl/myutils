from dataclasses import dataclass
import os
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import randperm
from torch.utils.data import DataLoader, TensorDataset

from ..config_base import ConfigBase
from ..file_name import FileName
from ..util_logger import BlankUtilLogger, UtilLogger

from .data_model import DatasetLike
from .lazy_loaded_dataset import LazyLoadedDataset


@dataclass
class _TypeAndShape:
    dtype: torch.dtype
    size: torch.Size


_TypeAndShapeLike = Union[_TypeAndShape, Tuple[_TypeAndShape, ...], Dict[Any, _TypeAndShape]]
_DataLike = Union[torch.Tensor, List[torch.Tensor], Dict[Any, torch.Tensor]]


def split_and_save_dataset(config: ConfigBase, base_dataset: DatasetLike,
                           test_rate: float, validation_rate: float, shuffle: bool,
                           file_batch_size: Optional[int] = None,
                           *, logger: UtilLogger) -> None:

    length: int = len(base_dataset)
    # assert all(length == tensor.shape[0] for tensor in base_dataset.tensors)

    shuffled: torch.Tensor = randperm(length) if shuffle else torch.arange(length)
    test_length: int = int(length * test_rate)
    validation_length: int = int((length - test_length) * validation_rate)
    train_length: int = length - (test_length + validation_length)

    logger.debug("Train Dataset を保存します。")
    _save_dataset(base_dataset, shuffled[: train_length],
                  config.processed_data_directory, config.train_file, file_batch_size, logger)
    offset: int = train_length
    logger.debug("Validation Dataset を保存します。")
    _save_dataset(base_dataset, shuffled[offset: offset + validation_length],
                  config.processed_data_directory, config.validation_file, file_batch_size, logger)
    offset += validation_length
    logger.debug("Test Dataset を保存します。")
    _save_dataset(base_dataset, shuffled[offset: offset + test_length],
                  config.processed_data_directory, config.test_file, file_batch_size, logger)


def _save_dataset(base_dataset: DatasetLike, indices: torch.Tensor, directory: str, file_name: FileName,
                  file_batch_size: Optional[int] = None,
                  logger: UtilLogger = BlankUtilLogger) -> None:

    if indices.size(0) == 0:
        return
    index_loader: DataLoader = DataLoader(
        TensorDataset(indices), batch_size=file_batch_size or indices.size(0), shuffle=False)
    type_and_shape: _TypeAndShapeLike = _type_and_shape(base_dataset[0])
    count: int = 0

    for file_batch_indices in index_loader:
        length: int = file_batch_indices[0].size(0)
        data: _DataLike = _init_batch_data(type_and_shape, length)
        # data: List[torch.Tensor] = [torch.empty((length, *tensor.shape[1:]), dtype=tensor.dtype)
        #                             for tensor in base_dataset[0]]
        for i in range(length):
            index: int = file_batch_indices[0][i].item()
            if isinstance(data, torch.Tensor):
                data[i] = base_dataset[index]
            if isinstance(data, Sequence):
                for j in range(len(data)):
                    data[j][i] = base_dataset[index][j]
            if isinstance(data, Mapping):
                for k in data.keys():
                    data[k][i] = base_dataset[index][k]
        dataset: TensorDataset = TensorDataset(*data)
        file_suffix: str = "" if file_batch_size is None else os.sep + str(count)
        if file_suffix:
            os.makedirs(directory + os.sep + file_name.name_without_ext, exist_ok=True)
        file_path: str = directory + os.sep + file_name.get_full_name(suffix=file_suffix)
        torch.save(dataset, file_path)
        # logger.debug("(count = {}, length = {}, shape = {}, path = {})".format(
        #     count + 1, length, str([t.shape for t in dataset.tensors]), file_path))
        logger.debug("(shape = {}, path = {})".format(str([t.shape for t in dataset.tensors]), file_path))
        count += 1


def _type_and_shape(data0: Any) -> _TypeAndShapeLike:
    if isinstance(data0, torch.Tensor):
        return _TypeAndShape(data0.dtype, data0.size())
    if isinstance(data0, Sequence):
        return tuple(_type_and_shape(d) for d in data0)
    if isinstance(data0, Mapping):
        return {k: _type_and_shape(d) for k, d in data0.items()}
    raise TypeError


def _init_batch_data(type_and_shape: _TypeAndShapeLike, length: int) -> Any:
    if isinstance(type_and_shape, _TypeAndShape):
        return torch.empty((length, *type_and_shape.size), dtype=type_and_shape.dtype)
    if isinstance(type_and_shape, Sequence):
        return [_init_batch_data(ts, length) for ts in type_and_shape]
    if isinstance(type_and_shape, Mapping):
        return {k: _init_batch_data(ts, length) for k, ts in type_and_shape.items()}
    raise TypeError


def exist_dataset_files(config: ConfigBase, file_name: FileName) -> bool:
    return os.path.isfile(config.processed_data_directory + os.sep + file_name.get_full_name()) or \
           os.path.isfile(config.processed_data_directory + os.sep + file_name.get_full_name(suffix=os.sep + str(0)))


def iterate_dataset_path_list(config: ConfigBase, file_name: FileName) -> Iterator[str]:

    if os.path.isfile(config.processed_data_directory + os.sep + file_name.get_full_name()):
        yield config.processed_data_directory + os.sep + file_name.get_full_name()
        return

    dataset_file_path: Callable[[int], bool] = \
        lambda x: config.processed_data_directory + os.sep + file_name.get_full_name(suffix=os.sep + str(x))

    # if os.path.isfile(dataset_file_path(0)):
    if os.path.isdir(config.processed_data_directory + os.sep + file_name.name_without_ext):
        count: int = 0
        while os.path.isfile(dataset_file_path(count)):
            yield dataset_file_path(count)
            count += 1
        return


def create_lazy_loaded_dataset(config: ConfigBase, file_name: FileName,
                               *, logger: UtilLogger = BlankUtilLogger) -> LazyLoadedDataset:
    return LazyLoadedDataset(tuple(iterate_dataset_path_list(config, file_name)), logger=logger)


def create_or_load_datasets(config: ConfigBase, base_dataset: DatasetLike, file_name: FileName,
                            test_rate: float, validation_rate: float, shuffle: bool,
                            file_batch_size: Optional[int] = None,
                            *, logger: UtilLogger = BlankUtilLogger) -> Iterator[TensorDataset]:

    file_path_list: Tuple[str, ...] = tuple(iterate_dataset_path_list(config, file_name))

    if len(file_path_list) == 0:
        split_and_save_dataset(config=config, base_dataset=base_dataset,
                               test_rate=test_rate, validation_rate=validation_rate, shuffle=shuffle,
                               file_batch_size=file_batch_size, logger=logger)
        file_path_list = tuple(iterate_dataset_path_list(config, file_name))

    for file_path in file_path_list:
        yield torch.load(file_path)


# def create_train_datasets(config: ConfigBase, base_dataset: TensorDatasetLike,
#                           test_rate: float, validation_rate: float, shuffle: bool,
#                           file_batch_size: Optional[int] = None,
#                           logger: UtilLogger = BlankUtilLogger) -> Iterator[TensorDataset]:
#     return create_or_load_datasets(config=config, base_dataset=base_dataset, file_name=config.train_file,
#                                    test_rate=test_rate, validation_rate=validation_rate, shuffle=shuffle,
#                                    file_batch_size=file_batch_size, logger=logger)
