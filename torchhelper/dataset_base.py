import os
from typing import Callable, Iterator, List, Optional, Sequence

import torch
from torch import randperm
from torch.utils.data import DataLoader, TensorDataset

from ..config_base import ConfigBase
from ..file_name import FileName
from ..util_logger import BlankUtilLogger, UtilLogger

from .data_model import TensorDatasetLike


def split_and_save_dataset(config: ConfigBase, base_dataset: TensorDatasetLike,
                           test_rate: float, validation_rate: float, shuffle: bool,
                           file_batch_size: Optional[int] = None,
                           *, logger: UtilLogger) -> None:
    if not issubclass(type(base_dataset), TensorDataset):
        return

    length: int = base_dataset.tensors[0].shape[0]
    # assert all(length == tensor.shape[0] for tensor in base_dataset.tensors)

    shuffled: torch.Tensor = randperm(length) if shuffle else torch.arange(length)
    test_length: int = int(length * test_rate)
    validation_length: int = int((length - test_length) * validation_rate)
    train_length: int = length - (test_length + validation_length)

    logger.debug("Train Dataset を保存します。")
    _save_dataset(base_dataset.tensors, shuffled[: train_length],
                  config.processed_data_directory, config.train_file, file_batch_size, logger)
    offset: int = train_length
    logger.debug("Validation Dataset を保存します。")
    _save_dataset(base_dataset.tensors, shuffled[offset: offset + validation_length],
                  config.processed_data_directory, config.validation_file, file_batch_size, logger)
    offset += validation_length
    logger.debug("Test Dataset を保存します。")
    _save_dataset(base_dataset.tensors, shuffled[offset: offset + test_length],
                  config.processed_data_directory, config.test_file, file_batch_size, logger)


def _save_dataset(tensors: Sequence[torch.Tensor], indices: torch.Tensor, directory: str, file_name: FileName,
                  file_batch_size: Optional[int] = None,
                  logger: UtilLogger = BlankUtilLogger) -> None:
    if indices.shape[0] == 0:
        return
    index_loader: DataLoader = DataLoader(
        TensorDataset(indices), batch_size=file_batch_size or indices.shape[0], shuffle=False)
    count: int = 0
    for file_batch_indices in index_loader:
        length: int = file_batch_indices[0].shape[0]
        data: List[torch.Tensor] = [torch.empty((length, *tensor.shape[1:]), dtype=tensor.dtype)
                                    for tensor in tensors]
        for i in range(length):
            index: int = file_batch_indices[0][i].item()
            for j in range(len(data)):
                data[j][i] = tensors[j][index]
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


def create_lazy_loaded_datasets(config: ConfigBase, file_name: FileName,
                                test_rate: float, validation_rate: float, shuffle: bool,
                                file_batch_size: Optional[int] = None,
                                *, logger: UtilLogger = BlankUtilLogger) -> Iterator[TensorDataset]:
    pass


def create_or_load_datasets(config: ConfigBase, base_dataset: TensorDatasetLike, file_name: FileName,
                            test_rate: float, validation_rate: float, shuffle: bool,
                            file_batch_size: Optional[int] = None,
                            *, logger: UtilLogger = BlankUtilLogger) -> Iterator[TensorDataset]:

    if os.path.isfile(config.processed_data_directory + os.sep + file_name.get_full_name()):
        yield torch.load(config.processed_data_directory + os.sep + file_name.get_full_name())
        return

    dataset_file_path: Callable[[int], bool] = \
        lambda x: config.processed_data_directory + os.sep + file_name.get_full_name(suffix=os.sep + str(x))

    # if os.path.isfile(dataset_file_path(0)):
    if os.path.isdir(config.processed_data_directory + os.sep + file_name.name_without_ext):
        count: int = 0
        while os.path.isfile(dataset_file_path(count)):
            yield torch.load(dataset_file_path(count))
            count += 1
        return

    split_and_save_dataset(config=config, base_dataset=base_dataset,
                           test_rate=test_rate, validation_rate=validation_rate, shuffle=shuffle,
                           file_batch_size=file_batch_size, logger=logger)

    for d in create_or_load_datasets(config=config, base_dataset=base_dataset, file_name=file_name,
                                     test_rate=test_rate, validation_rate=validation_rate, shuffle=shuffle,
                                     file_batch_size=file_batch_size, logger=logger):
        yield d


def create_train_datasets(config: ConfigBase, base_dataset: TensorDatasetLike,
                          test_rate: float, validation_rate: float, shuffle: bool,
                          file_batch_size: Optional[int] = None,
                          logger: UtilLogger = BlankUtilLogger) -> Iterator[TensorDataset]:
    return create_or_load_datasets(config=config, base_dataset=base_dataset, file_name=config.train_file,
                                   test_rate=test_rate, validation_rate=validation_rate, shuffle=shuffle,
                                   file_batch_size=file_batch_size, logger=logger)
