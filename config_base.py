import os
from typing import Any, Callable, Dict, Generic, Iterable, Optional, Tuple, Type, TypeVar

from optuna.pruners import BasePruner
from torch import nn
from torch.optim.optimizer import Optimizer

from .file_name import FileName
from .io import get_common_root_directory

from .exceptions import DuplicateModuleArgumentsError

T = TypeVar("T")


class ModuleConfig(Generic[T]):
    # noinspection PyUnusedLocal
    def __init__(self, module_type: Type[T], module_parameter: Iterable[Tuple[str, Any]] = (), *args, **kwargs):
        self.type: Type[T] = module_type
        self.parameters: Dict[str, Any] = {}
        for key, value in module_parameter:
            if key in self.parameters:
                raise DuplicateModuleArgumentsError
            self.parameters[key] = value
        for key, value in kwargs.items():
            if key in self.parameters:
                raise DuplicateModuleArgumentsError
            self.parameters[key] = value


class ModelSetConfig:
    # noinspection PyUnusedLocal
    def __init__(self, model: ModuleConfig[nn.Module],
                 criterion: ModuleConfig[nn.Module], optimizer: ModuleConfig[Optimizer],
                 *args, **kwargs):
        self.model: ModuleConfig[nn.Module] = model
        self.criterion: ModuleConfig[nn.Module] = criterion
        self.optimizer: ModuleConfig[Optimizer] = optimizer


class TrainConfig:

    # noinspection PyUnusedLocal
    def __init__(self, batch_size: int, number_of_epochs: int, progress_epoch: int, temporary_save_epoch: int,
                 pre_epoch: Optional[int] = None, warmup_progress_epoch: int = 0,
                 optuna_number_of_trials: Optional[int] = None, optuna_pruner: Optional[BasePruner] = None,
                 optuna_pre_trials: int = 0, *args, **kwargs):

        self.batch_size: int = batch_size
        self.number_of_epochs: int = number_of_epochs

        self.progress_epoch: int = progress_epoch
        self.temporary_save_epoch: int = temporary_save_epoch

        self.pre_epoch: Optional[int] = pre_epoch
        self.warmup_progress_epoch: int = warmup_progress_epoch

        self.optuna_pruner: Optional[BasePruner] = optuna_pruner
        self.optuna_number_of_trials: Optional[int] = optuna_number_of_trials
        self.optuna_pre_trials: int = optuna_pre_trials


class ConfigBase:

    # noinspection PyUnusedLocal
    def __init__(self, model: ModuleConfig[nn.Module],
                 criterion: ModuleConfig[nn.Module], optimizer: ModuleConfig[Optimizer],
                 train: TrainConfig, *args, root_dir: str = "", **kwargs):

        class_name: str = self.__class__.__name__
        root_dir = str(get_common_root_directory(root_dir)) or "."

        self.log_directory: str = root_dir + os.sep + "log" + os.sep + class_name
        self.interim_directory: str = root_dir + os.sep + "interim" + os.sep + class_name
        self.result_directory: str = root_dir + os.sep + "results" + os.sep + class_name

        self.raw_data_directory: str = root_dir + os.sep + "data" + os.sep + "raw"
        self.processed_data_directory: str = root_dir + os.sep + "data" + os.sep + "processed" + os.sep + class_name

        self.model_file: FileName = FileName("model", "pth")
        self.loss_file: FileName = FileName("loss", "csv")
        self.score_file: FileName = FileName("score", "csv")

        self.optuna_score_file: FileName = FileName("optuna_score", "csv")
        self.optuna_params_file: FileName = FileName("optuna_params", "json5")

        self.get_epoch_str_function: Callable[[int], str] \
            = lambda epoch: "(epoch={})".format(epoch) if epoch >= 0 else ""

        self.model_set: ModelSetConfig = ModelSetConfig(model=model, criterion=criterion, optimizer=optimizer)
        self.train: TrainConfig = train
