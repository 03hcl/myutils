import os
from typing import Callable, Iterator, List, Optional, Sequence

import torch
from torch import randperm
from torch.utils.data import DataLoader, TensorDataset

from ..config_base import ConfigBase
from ..file_name import FileName
from ..util_logger import BlankUtilLogger, UtilLogger


def split_and_save_dataset(config: ConfigBase, tensors: Sequence[torch.Tensor],
                           test_rate: float, validation_rate: float, shuffle: bool,
                           file_batch_size: Optional[int] = None,
                           *, logger: UtilLogger) -> None:

    length: int = tensors[0].shape[0]
    assert all(length == tensor.shape[0] for tensor in tensors)

    shuffled: torch.Tensor = randperm(length) if shuffle else torch.arange(length)
    test_length: int = int(length * test_rate)
    validation_length: int = int((length - test_length) * validation_rate)
    train_length: int = length - (test_length + validation_length)

    logger.debug("Train Dataset を保存します。")
    _save_dataset(tensors, shuffled[: train_length],
                  config.processed_data_directory, config.train_file, file_batch_size, logger)
    offset: int = train_length
    logger.debug("Validation Dataset を保存します。")
    _save_dataset(tensors, shuffled[offset: offset + validation_length],
                  config.processed_data_directory, config.validation_file, file_batch_size, logger)
    offset += validation_length
    logger.debug("Test Dataset を保存します。")
    _save_dataset(tensors, shuffled[offset: offset + test_length],
                  config.processed_data_directory, config.test_file, file_batch_size, logger)


def _save_dataset(tensors: Sequence[torch.Tensor], indices: torch.Tensor, directory: str, file_name: FileName,
                  file_batch_size: Optional[int] = None,
                  logger: UtilLogger = BlankUtilLogger) -> None:
    file_batch_size = file_batch_size or indices.shape[0]
    if file_batch_size == 0:
        return
    index_loader: DataLoader = DataLoader(
        TensorDataset(indices), batch_size=file_batch_size, shuffle=False)
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
        file_path: str = directory + os.sep + file_name.get_full_name(suffix=str(count))
        torch.save(dataset, file_path)
        # logger.debug("(count = {}, length = {}, shape = {}, path = {})".format(
        #     count + 1, length, str([t.shape for t in dataset.tensors]), file_path))
        logger.debug("(shape = {}, path = {})".format(str([t.shape for t in dataset.tensors]), file_path))
        count += 1


def create_or_load_datasets(config: ConfigBase, tensors: Sequence[torch.Tensor], file_name: FileName,
                            test_rate: float, validation_rate: float, shuffle: bool,
                            file_batch_size: Optional[int] = None,
                            *, logger: UtilLogger = BlankUtilLogger) -> Iterator[TensorDataset]:

    if os.path.isfile(config.processed_data_directory + os.sep + file_name.get_full_name()):
        yield torch.load(config.processed_data_directory + os.sep + file_name.get_full_name())
        return

    exists_dataset_file: Callable[[int], bool] = lambda x: os.path.isfile(
        config.processed_data_directory + os.sep + file_name.get_full_name(suffix=str(x)))

    if not exists_dataset_file(0):
        split_and_save_dataset(config=config, tensors=tensors,
                               test_rate=test_rate, validation_rate=validation_rate, shuffle=shuffle,
                               file_batch_size=file_batch_size, logger=logger)

    count: int = 0
    while exists_dataset_file(count):
        yield torch.load(config.processed_data_directory + os.sep + file_name.get_full_name(suffix=str(count)))
        count += 1


def create_train_datasets(config: ConfigBase, tensors: Sequence[torch.Tensor],
                          test_rate: float, validation_rate: float, shuffle: bool,
                          file_batch_size: Optional[int] = None,
                          logger: UtilLogger = BlankUtilLogger) -> Iterator[TensorDataset]:
    return create_or_load_datasets(config=config, tensors=tensors, file_name=config.train_file,
                                   test_rate=test_rate, validation_rate=validation_rate, shuffle=shuffle,
                                   file_batch_size=file_batch_size, logger=logger)
