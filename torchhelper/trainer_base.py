from logging import Logger
import os
import shutil
from typing import Any, Dict, Optional, Tuple

import numpy as np

import optuna
from optuna.exceptions import TrialPruned
from optuna.study import Study
from optuna.trial import Trial

import torch
from torch.utils.data import DataLoader

from .. import ConfigBase, UtilLogger
from ..visualizer import visualize_loss

from .both_phase_base import create_data_loader_dict, create_key_str, get_data_length
from .device import Device
from .data_model import DataLoaderLike, DatasetLike, DataTensorLike, NumericScore, Score, ScoreLike, \
    EpochResult, TrainLog, TrainResult, TrainResultOfDataLoader, \
    TRAIN_KEY_STR, VALIDATION_KEY_STR
from .model_set import create_model_set, load_interim_model_set, ModelSet


class TrainerBase:

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
    def train_for_each_iteration(cls, model_set: ModelSet, data: DataTensorLike, backpropagate: bool,
                                 *, logger: UtilLogger, **kwargs) -> TrainResult:
        raise NotImplementedError

    # noinspection PyUnusedLocal
    @classmethod
    def score_for_each_iteration(cls, model_set: ModelSet, data: DataTensorLike, train_result: TrainResult,
                                 *, logger: UtilLogger, **kwargs) -> Optional[ScoreLike]:
        return None

    # noinspection PyUnusedLocal
    @classmethod
    def output_progress(cls, config: ConfigBase, epoch: int, model_set: ModelSet,
                        epoch_result_dict: Dict[str, EpochResult], train_keys: Tuple[str],
                        loss_array: np.ndarray, visualizes_loss_on_logscale: bool = False,
                        *, logger: UtilLogger, **kwargs) -> None:
        score: Optional[float] = calculate_score_sum(train_keys, epoch_result_dict)
        log_prefix: str = "" if score is None else "[Score = {:10.6f}]".format(score)
        logger.snap_epoch_with_loss(calculate_loss_sum(train_keys, epoch_result_dict),
                                    epoch, config.train.number_of_epochs, pre_epoch=config.train.pre_epoch or 0,
                                    log_prefix=log_prefix)
        visualize_loss(loss_array, tuple(epoch_result_dict.keys()),
                       file_name="loss" + config.get_epoch_str_function(epoch),
                       pre_epoch=config.train.pre_epoch, directory=config.interim_directory,
                       is_logscale=visualizes_loss_on_logscale)

    @classmethod
    def train(cls, config: ConfigBase, model_set: ModelSet,
              train_dataset: DatasetLike, validation_dataset: Optional[DatasetLike],
              visualizes_loss_on_logscale: bool = False,
              *, logger: UtilLogger, trial: Optional[Trial] = None, **kwargs) -> TrainLog:

        pre_epoch: int = config.train.pre_epoch or 0

        data_loader_like: DataLoaderLike = \
            cls.create_data_loader(config=config, dataset=train_dataset, logger=logger, **kwargs)
        data_loader_dict: Dict[str, DataLoader] = \
            create_data_loader_dict(data_loader_like, create_key_str(TRAIN_KEY_STR))
        train_keys: Tuple[str] = tuple(data_loader_dict.keys())

        if validation_dataset is not None:
            data_loader_like = \
                cls.create_data_loader(config=config, dataset=validation_dataset, logger=logger, **kwargs)
            data_loader_dict.update(create_data_loader_dict(data_loader_like, create_key_str(VALIDATION_KEY_STR)))

        data_loader_length: int = len(data_loader_dict)

        loss_array_length: int = \
            int(config.train.number_of_epochs / config.train.progress_epoch) \
            - int(max(pre_epoch, config.train.warmup_progress_epoch) / config.train.progress_epoch)
        loss_array: np.ndarray = np.empty((loss_array_length, data_loader_length + 1))
        loss_index: int = 0

        logger.time_meter.set_start_time()

        # noinspection PyUnusedLocal
        result_dict: Dict[str, EpochResult] = dict()
        # noinspection PyUnusedLocal
        epoch: int

        for epoch in range(pre_epoch, config.train.number_of_epochs):

            result_dict = dict()

            is_output_progress: bool \
                = config.train.progress_epoch > 0 \
                and epoch >= config.train.warmup_progress_epoch \
                and (epoch + 1) % config.train.progress_epoch == 0

            # noinspection PyUnusedLocal
            key: str
            # noinspection PyUnusedLocal
            data_loader: DataLoader

            for key, data_loader in data_loader_dict.items():
                result: TrainResultOfDataLoader = cls._train_for_each_data_loader(
                    model_set=model_set, data_loader=data_loader,
                    is_output_progress=is_output_progress, backpropagate=(key in train_keys), logger=logger, **kwargs)
                if issubclass(type(result.loss), float):
                    result.loss /= result.data_count
                if issubclass(type(result.score), float):
                    result.score /= result.data_count
                result_dict[key] = EpochResult(data_loader, result.data_count, result.loss, result.score)

            if is_output_progress:
                loss_array[loss_index, :] = [epoch + 1, *(e.loss for e in result_dict.values())]
                loss_index += 1
                cls.output_progress(
                    config=config, epoch=epoch + 1,
                    model_set=model_set, epoch_result_dict=result_dict, train_keys=train_keys,
                    loss_array=loss_array[: loss_index], visualizes_loss_on_logscale=visualizes_loss_on_logscale,
                    logger=logger, **kwargs)

            if config.train.temporary_save_epoch > 0 and (epoch + 1) % config.train.temporary_save_epoch == 0:
                logger.save_loss_array_and_model(config, epoch + 1, loss_array[0: loss_index, :], model_set.model,
                                                 trial=trial)

            if config.train.optuna_pruner is not None and trial is not None:
                loss_sum: float = calculate_loss_sum(train_keys, result_dict)
                trial.report(loss_sum, epoch + 1)
                if trial.should_prune(epoch + 1):
                    logger.info("epoch = {:>5},    loss = {:>10.6f},    [Pruned]".format(epoch + 1, loss_sum))
                    raise TrialPruned

        return TrainLog(tuple(data_loader_dict.keys()), loss_array, calculate_loss_sum(train_keys, result_dict))

    @classmethod
    def _train_for_each_data_loader(cls, model_set: ModelSet, data_loader: DataLoader,
                                    is_output_progress: bool, backpropagate: bool,
                                    *, logger: UtilLogger, **kwargs) -> TrainResultOfDataLoader:

        if backpropagate:
            model_set.model.train()
        else:
            model_set.model.eval()

        loss_sum: NumericScore = 0
        score_sum: Optional[Score] = 0
        data_count: int = 0

        # noinspection PyUnusedLocal
        data: DataTensorLike

        for data in data_loader:

            data_length: int = get_data_length(data)
            data_count += data_length

            # noinspection PyUnusedLocal
            result: TrainResult

            if backpropagate:
                with torch.autograd.detect_anomaly():
                    model_set.optimizer.zero_grad()
                    result = cls.train_for_each_iteration(
                        model_set=model_set, data=data, backpropagate=backpropagate, logger=logger, **kwargs)
                    model_set.optimizer.step()
            else:
                with torch.no_grad():
                    result = cls.train_for_each_iteration(
                        model_set=model_set, data=data, backpropagate=backpropagate, logger=logger, **kwargs)

            if issubclass(type(result.loss), int) or issubclass(type(result.loss), float):
                loss_sum += result.loss * data_length
            else:
                raise NotImplementedError

            if is_output_progress:
                score: Optional[ScoreLike] = cls.score_for_each_iteration(
                        model_set=model_set, data=data, train_result=result, logger=logger, **kwargs)
                if score is None:
                    score_sum = None
                elif issubclass(type(score), int) or issubclass(type(score), float):
                    score_sum += score * data_length
                elif issubclass(type(score), torch.Tensor):
                    if issubclass(type(score_sum), torch.Tensor):
                        score_sum = torch.cat((score_sum, score), dim=0)
                    else:
                        score_sum = score
                else:
                    raise NotImplementedError

        return TrainResultOfDataLoader(data_count, loss_sum, score_sum)

    @classmethod
    def run(cls, config: ConfigBase,
            *, visualizes_loss_on_logscale: bool = False,
            device: Optional[Device] = None, logger: Optional[Logger] = None,
            device_kwargs: Optional[Dict[str, Any]] = None, **kwargs) -> None:

        util_logger: UtilLogger = UtilLogger(config=config, base_logger=logger, file_name="train")
        device: Device = device or Device(**(device_kwargs or {}), logger=util_logger)
        train_dataset: DatasetLike = cls.create_dataset(config=config, logger=util_logger, **kwargs)
        validation_dataset: DatasetLike = cls.create_validation_dataset(config=config, logger=util_logger, **kwargs)

        def objective(trial: Optional[Trial] = None) -> float:

            if trial is not None:
                util_logger.info("Trial {:d}".format(trial.number + 1))

            # noinspection PyUnusedLocal
            model_set: ModelSet

            if config.train.pre_epoch is None:
                model_set = create_model_set(config=config, device=device, trial=trial, logger=util_logger)
            else:
                model_set = load_interim_model_set(config=config, device=device, trial=trial, logger=util_logger)

            try:
                train_log: TrainLog = cls.train(config=config, model_set=model_set,
                                                train_dataset=train_dataset, validation_dataset=validation_dataset,
                                                trial=trial, logger=util_logger, **kwargs)
            except TrialPruned as e:
                util_logger.info("訓練が途中で打ち切られました。")
                raise e
            except Exception as e:
                util_logger.warning("エラーが発生しました。")
                util_logger.warning(str(e.args))
                raise e
            model_set.model.eval()

            result_directory: str = config.result_directory
            if trial is not None:
                result_directory += os.sep + str(trial.number + 1)
            os.makedirs(result_directory, exist_ok=True)
            util_logger.debug("結果を保存します。(ディレクトリ = {})".format(result_directory))

            torch.save(model_set.model.state_dict(), result_directory + os.sep + config.model_file.full_name)
            # noinspection PyTypeChecker
            np.savetxt(result_directory + os.sep + config.loss_file.full_name, train_log.loss_array,
                       fmt="%.18e", delimiter=",")
            # logger_wrapper.save_loss_array_and_model(-1, loss_array, model_set.model,
            #                                          dir_path=config.result_directory)
            visualize_loss(train_log.loss_array, train_log.data_keys,
                           pre_epoch=config.train.pre_epoch, directory=result_directory,
                           is_logscale=visualizes_loss_on_logscale)

            return train_log.score

        util_logger.info("訓練を開始します。")

        if config.train.optuna_number_of_trials:

            study: Study = optuna.create_study(pruner=config.train.optuna_pruner)
            study.optimize(objective, n_trials=config.train.optuna_number_of_trials)

            util_logger.info("Best: Trial {:d} / Score = {:f}".format(study.best_trial.number + 1, study.best_value))
            util_logger.info("Parameters:")
            for name, value in study.best_params.items():
                if issubclass(type(value), float):
                    util_logger.info("{} = {:e}".format(name, value))
                elif issubclass(type(value), int):
                    util_logger.info("{} = {:d}".format(name, value))
                else:
                    util_logger.info("{} = {}".format(name, str(value)))

            best_result_directory: str = config.result_directory + os.sep + str(study.best_trial.number + 1)
            shutil.copy(best_result_directory + os.sep + config.model_file.full_name, config.result_directory)
            shutil.copy(best_result_directory + os.sep + config.loss_file.full_name, config.result_directory)
            shutil.copy(best_result_directory + os.sep + config.loss_file.name_without_ext + os.extsep + "png",
                        config.result_directory)

        else:
            objective()

        util_logger.close()


def calculate_loss_sum(train_keys: Tuple[str], epoch_result_dict: Dict[str, EpochResult]) -> float:
    data_sum: int = sum(result.data_count for key, result in epoch_result_dict.items() if key in train_keys)
    loss_sum: float = sum(result.loss for key, result in epoch_result_dict.items() if key in train_keys)
    return loss_sum / data_sum if data_sum > 0 else 0


def calculate_score_sum(train_keys: Tuple[str], epoch_result_dict: Dict[str, EpochResult]) -> Optional[float]:
    data_sum: int = sum(result.data_count for key, result in epoch_result_dict.items()
                        if ((key in train_keys) and (result.score is not None)))
    score_sum: float = sum(result.loss for key, result in epoch_result_dict.items()
                           if ((key in train_keys) and (result.score is not None)))
    return score_sum / data_sum if data_sum > 0 else None
