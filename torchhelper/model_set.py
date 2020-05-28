import json5
import os
from typing import Any, Dict, Optional

from optuna.trial import Trial

import torch
from torch import nn
from torch.optim.optimizer import Optimizer

from .. import ConfigBase, UtilLogger

from .device import Device
from .optuna_parameter import OptunaParameter

from .exceptions import CouldNotSuggestOptunaParameterError


class ModelSet:
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: Optimizer, device: Device):
        if device.is_gpu:
            model = model.to(device=device.torch_device)
            # criterion = criterion.to(device=device.torch_device)
        self.model: nn.Module = model
        self.criterion: nn.Module = criterion
        self.optimizer: Optimizer = optimizer
        self.device: Device = device


def create_model_set(config: ConfigBase, device: Device, *,
                     trial: Optional[Trial] = None, initial_model: nn.Module = None,
                     logger: Optional[UtilLogger] = None) -> ModelSet:

    model_parameters: Dict[str, Any] = _suggest_parameters(config.model_set.model.parameters, trial, config, logger)

    model: nn.Module = initial_model or config.model_set.model.type(**model_parameters)
    criterion: nn.Module = config.model_set.criterion.type(
        **_suggest_parameters(config.model_set.criterion.parameters, trial, config, logger))

    if logger is not None:
        param_str: str = ""
        for key, value in model_parameters.items():
            param_str += ", " if param_str else "("
            param_str += "{} = {}".format(key, str(value))
        if param_str:
            param_str += ")"
        logger.debug("モデル ({}) を生成します。{}".format(model.__class__.__name__, param_str))

    model_set = ModelSet(model, criterion,
                         config.model_set.optimizer.type(
                             params=model.parameters(),
                             **_suggest_parameters(config.model_set.optimizer.parameters, trial, config, logger)),
                         device)

    return model_set


def _suggest_parameters(parameters: Dict[str, Any], trial: Optional[Trial], config: ConfigBase,
                        logger: Optional[UtilLogger]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    params: Optional[Dict[str, Any]] = None
    for key, value in parameters.items():
        if not issubclass(type(value), OptunaParameter):
            result[key] = value
        elif trial is not None:
            optuna_parameter: OptunaParameter = value
            suggested_value = optuna_parameter.get_optuna_parameter(trial)
            if logger is not None:
                logger.info("{} = {}".format(optuna_parameter.name, str(suggested_value)))
            result[key] = suggested_value
        else:
            if params is None:
                params_file_path: str = config.result_directory + os.sep + config.optuna_params_file.full_name
                if not os.path.isfile(params_file_path):
                    raise CouldNotSuggestOptunaParameterError
                with open(params_file_path, "r") as f:
                    params = json5.load(f)
            result[key] = params[key]
    return result


def load_model_set(config: ConfigBase, device: Device, model_path: str,
                   *, trial: Optional[Trial] = None, logger: Optional[UtilLogger] = None) -> ModelSet:
    model_set = create_model_set(config, device, trial=trial, logger=logger)
    if logger is not None:
        logger.debug("パラメーターを読み込みます。(model_path = {})".format(model_path))
    model_set.model.load_state_dict(torch.load(model_path))
    return model_set


def load_result_model_set(config: ConfigBase, device: Device,
                          *, logger: Optional[UtilLogger] = None) -> ModelSet:
    return load_model_set(config, device,
                          model_path=config.result_directory + os.sep + config.model_file.full_name,
                          logger=logger)


def load_interim_model_set(config: ConfigBase, device: Device,
                           *, trial: Optional[Trial] = None, logger: Optional[UtilLogger] = None) -> ModelSet:
    return load_model_set(config, device,
                          model_path=config.interim_directory + os.sep + config.model_file.get_full_name(
                              suffix=config.get_epoch_str_function(config.train.pre_epoch)),
                          trial=trial, logger=logger)


