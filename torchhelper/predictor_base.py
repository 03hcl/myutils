from logging import Logger
from typing import Any, Dict, Optional, Union

import torch
from torch.utils.data import DataLoader

from .. import ConfigBase, UtilLogger

from .both_phase_base import create_data_loader_dict, create_key_str, get_data_length
from .data_model import DataLoaderLike, DatasetLike, DataTensorLike
from .device import Device
from .model_set import load_result_model_set, ModelSet


class PredictorBase:

    @classmethod
    def create_dataset(cls, config: ConfigBase, *, logger: UtilLogger, **kwargs) -> DatasetLike:
        raise NotImplementedError

    # noinspection PyUnusedLocal
    @classmethod
    def create_validation_dataset(cls, config: ConfigBase, *, logger: UtilLogger, **kwargs) -> Optional[DatasetLike]:
        return None

    @classmethod
    def create_data_loader(cls, config: ConfigBase, dataset: DatasetLike, *, logger: UtilLogger, **kwargs) \
            -> DataLoaderLike:
        raise NotImplementedError

    @classmethod
    def predict_for_each_iteration(cls, model_set: ModelSet, data: DataTensorLike, *, logger: UtilLogger, **kwargs) \
            -> DataTensorLike:
        raise NotImplementedError

    # noinspection PyUnusedLocal
    @classmethod
    def predict(cls, config: ConfigBase, model_set: ModelSet,
                test_dataset: DatasetLike, validation_dataset: Optional[DatasetLike],
                *, logger: Optional[UtilLogger] = None, **kwargs) -> Dict[str, DataTensorLike]:

        data_loader_like: DataLoaderLike = \
            cls.create_data_loader(config=config, dataset=test_dataset, logger=logger, **kwargs)
        data_loader_dict: Dict[str, DataLoader] = create_data_loader_dict(data_loader_like, create_key_str("test"))

        if validation_dataset is not None:
            data_loader_like = \
                cls.create_data_loader(config=config, dataset=validation_dataset, logger=logger, **kwargs)
            data_loader_dict.update(create_data_loader_dict(data_loader_like, create_key_str("val")))

        output: Dict[str, DataTensorLike] = dict()

        # noinspection PyUnusedLocal
        key: str
        # noinspection PyUnusedLocal
        data_loader: DataLoader

        for key, data_loader in data_loader_dict.items():
            output[key] = cls._predict_for_each_data_loader(data_loader=data_loader, model_set=model_set, logger=logger)

        return output

    @classmethod
    def _predict_for_each_data_loader(cls, model_set: ModelSet, data_loader: DataLoader, *, logger: UtilLogger) \
            -> DataTensorLike:

        data_count: int = 0
        output: Optional[DataTensorLike] = None

        # noinspection PyUnusedLocal
        data: DataTensorLike

        for data in data_loader:
            data_length: int = get_data_length(data)
            data_count += data_length
            with torch.no_grad():
                batch_output: DataTensorLike = \
                    cls.predict_for_each_iteration(model_set=model_set, data=data, logger=logger)
            output = _init_batch(batch_output, data_length) if output is None \
                else _concat_batch(output, batch_output, data_length)

        return _mean_batch(output, data_count)

    @classmethod
    def run(cls, config: ConfigBase,
            *, device: Optional[Device] = None, logger: Optional[Logger] = None,
            device_kwargs: Optional[Dict[str, Any]] = None, **kwargs) -> ModelSet:

        util_logger: UtilLogger = UtilLogger(config=config, base_logger=logger, file_name="predict")
        device: Device = device or Device(**(device_kwargs or {}), logger=util_logger)
        test_dataset: DatasetLike = cls.create_dataset(config=config, logger=util_logger, **kwargs)
        validation_dataset: Optional[DatasetLike] = \
            cls.create_validation_dataset(config=config, logger=util_logger, **kwargs)

        model_set: ModelSet = load_result_model_set(config=config, device=device, logger=util_logger)
        model_set.model.eval()
        output: Dict[str, DataTensorLike] = \
            cls.predict(config=config, model_set=model_set,
                        test_dataset=test_dataset, validation_dataset=validation_dataset, logger=util_logger, **kwargs)
        cls.run_append(config=config, model_set=model_set, dataset=test_dataset, validation_dataset=validation_dataset,
                       output=output, logger=util_logger, **kwargs)

        util_logger.close()

        return model_set

    @classmethod
    def run_append(cls, config: ConfigBase, model_set: ModelSet,
                   dataset: DatasetLike, validation_dataset: Optional[DatasetLike], output: Dict[str, DataTensorLike],
                   *, device: Optional[Device] = None, logger: Optional[UtilLogger] = None, **kwargs) -> None:
        pass


def _concat_batch(base: DataTensorLike, appended: Union[DataTensorLike], data_length: int) -> DataTensorLike:
    if issubclass(type(appended), torch.Tensor):
        return _cat_or_sum(base, appended, data_length)
    if issubclass(type(appended), tuple) or issubclass(type(appended), list) or issubclass(type(appended), set):
        return tuple(_cat_or_sum(b, a, data_length) for b, a in zip(base, appended))
    if issubclass(type(appended), dict):
        for k, v in appended.items():
            base[k] = _cat_or_sum(base[k], v, data_length)
        return base
    raise TypeError


def _cat_or_sum(base: torch.Tensor, appended: torch.Tensor, data_length: int) -> torch.Tensor:
    return torch.cat((base, appended), dim=0) \
        if (appended.dim() > 0 and appended.shape[0] == data_length) \
        else base + appended * data_length


def _mean_batch(output: DataTensorLike, data_length: int) -> DataTensorLike:
    if issubclass(type(output), torch.Tensor):
        return output / data_length if output.dim() == 0 else output
    if issubclass(type(output), tuple) or issubclass(type(output), list) or issubclass(type(output), set):
        for o in output:
            if o.dim() == 0:
                o /= data_length
        return output
    if issubclass(type(output), dict):
        for o in output.values():
            if o.dim() == 0:
                o /= data_length
        return output
    raise TypeError


def _init_batch(output: DataTensorLike, data_length: int) -> DataTensorLike:
    if issubclass(type(output), torch.Tensor):
        return output * data_length if output.dim() == 0 else output
    if issubclass(type(output), tuple) or issubclass(type(output), list) or issubclass(type(output), set):
        for o in output:
            if o.dim() == 0:
                o *= data_length
        return output
    if issubclass(type(output), dict):
        for o in output.values():
            if o.dim() == 0:
                o *= data_length
        return output
    raise TypeError
