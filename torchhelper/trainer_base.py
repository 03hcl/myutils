import json5
from logging import Logger
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import optuna
from optuna.exceptions import TrialPruned
from optuna.study import Study
from optuna.trial import Trial

import torch
from torch.utils.data import DataLoader

from .. import ConfigBase, UtilLogger, is_debugging
from ..torchhelper import OptunaParameter
from ..visualizer import visualize_loss, visualize_values_for_each_epoch

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
                        epoch_result_dict: Dict[str, EpochResult],
                        train_keys: Tuple[str, ...], validation_keys: Tuple[str, ...],
                        loss_array: np.ndarray, score_array: Optional[np.ndarray] = None,
                        visualizes_loss_on_logscale: bool = False, *, logger: UtilLogger, **kwargs) -> None:
        score: Optional[float] = calculate_score_sum(train_keys, epoch_result_dict)
        log_prefix: str = "" if score is None else "[Score = {:10.6f}]".format(score)
        logger.snap_epoch_with_loss(calculate_loss_sum(train_keys, epoch_result_dict),
                                    epoch, config.train.number_of_epochs, pre_epoch=config.train.pre_epoch or 0,
                                    log_prefix=log_prefix)
        visualize_loss(loss_array, tuple(epoch_result_dict.keys()),
                       directory=config.interim_directory, file_name="loss" + config.get_epoch_str_function(epoch),
                       pre_epoch=config.train.pre_epoch, is_logscale=visualizes_loss_on_logscale)
        if score_array is not None:
            visualize_values_for_each_epoch(
                score_array, tuple(epoch_result_dict.keys()),
                directory=config.interim_directory, file_name="score" + config.get_epoch_str_function(epoch),
                pre_epoch=config.train.pre_epoch, y_axis_name="score", is_logscale=visualizes_loss_on_logscale)

    # noinspection PyUnusedLocal
    @classmethod
    def calculate_score(cls, train_keys: Tuple[str, ...], validation_keys: Tuple[str, ...],
                        epoch_result_dict: Dict[str, EpochResult], *, logger: UtilLogger, **kwargs) -> float:
        return calculate_loss_sum(train_keys if len(validation_keys) == 0 else validation_keys, epoch_result_dict)

    @classmethod
    def train(cls, config: ConfigBase, model_set: ModelSet,
              train_dataset: DatasetLike, validation_dataset: Optional[DatasetLike],
              visualizes_loss_on_logscale: bool = False,
              *, logger: UtilLogger, trial: Optional[Trial] = None, **kwargs) -> TrainLog:

        best_model_path: str = config.interim_directory + os.sep + config.model_file.full_name
        pre_epoch: int = config.train.pre_epoch or 0

        # region data_loader_dict の作成

        data_loader_like: DataLoaderLike = \
            cls.create_data_loader(config=config, dataset=train_dataset, logger=logger, **kwargs)
        data_loader_dict: Dict[str, DataLoader] = \
            create_data_loader_dict(data_loader_like, create_key_str(TRAIN_KEY_STR))

        train_keys: Tuple[str, ...] = tuple(data_loader_dict.keys())
        validation_keys: Tuple[str, ...] = tuple()

        if validation_dataset is not None:
            validation_data_loader_like: DataLoaderLike = \
                cls.create_data_loader(config=config, dataset=validation_dataset, logger=logger, **kwargs)
            validation_data_loader_dict: Dict[str, DataLoader] = \
                create_data_loader_dict(validation_data_loader_like, create_key_str(VALIDATION_KEY_STR))
            validation_keys = tuple(validation_data_loader_dict.keys())
            data_loader_dict.update(validation_data_loader_dict)

        data_loader_length: int = len(data_loader_dict)

        # endregion

        loss_array_length: int = \
            int(config.train.number_of_epochs / config.train.progress_epoch) \
            - int(max(pre_epoch, config.train.warmup_progress_epoch) / config.train.progress_epoch)
        loss_array: np.ndarray = np.empty((loss_array_length, data_loader_length + 1))
        score_array: np.ndarray = np.zeros((loss_array_length, data_loader_length + 1))
        loss_index: int = 0

        optuna_score_array: np.ndarray = np.zeros((config.train.number_of_epochs - pre_epoch, 2))

        latest_score: float = np.inf
        optuna_latest_score: float = np.inf

        # noinspection PyUnusedLocal
        result_dict: Dict[str, EpochResult] = dict()
        # noinspection PyUnusedLocal
        epoch: int

        logger.time_meter.set_start_time()

        for epoch in range(pre_epoch, config.train.number_of_epochs):

            result_dict = dict()

            is_output_progress: bool \
                = config.train.progress_epoch > 0 \
                and epoch >= config.train.warmup_progress_epoch \
                and (epoch + 1) % config.train.progress_epoch == 0
            has_score: bool = False

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
                has_score |= result.score is not None
                result_dict[key] = EpochResult(data_loader, result.data_count, result.loss, result.score)

            score: float = cls.calculate_score(train_keys, validation_keys, result_dict, logger=logger)

            if is_output_progress:
                loss_array[loss_index, :] = [epoch + 1, *(e.loss for e in result_dict.values())]
                if has_score:
                    score_array[loss_index, :] = [epoch + 1, *(e.score for e in result_dict.values())]
                loss_index += 1
                cls.output_progress(
                    config=config, epoch=epoch + 1,
                    model_set=model_set, epoch_result_dict=result_dict,
                    train_keys=train_keys, validation_keys=validation_keys,
                    loss_array=loss_array[: loss_index], score_array=score_array[: loss_index] if has_score else None,
                    visualizes_loss_on_logscale=visualizes_loss_on_logscale,
                    logger=logger, **kwargs)
                if score < latest_score:
                    latest_score = score
                    torch.save(model_set.model.state_dict(), best_model_path)
                    logger.info("モデル保存: path = {}".format(best_model_path))

            if config.train.temporary_save_epoch > 0 and (epoch + 1) % config.train.temporary_save_epoch == 0:
                logger.save_loss_array_and_model(config, epoch + 1, loss_array[0: loss_index, :], model_set.model,
                                                 trial=trial)

            if config.train.optuna_pruner is not None and trial is not None:
                optuna_latest_score = min(optuna_latest_score, score)
                trial.report(score, epoch + 1)
                optuna_score_array[epoch - pre_epoch, 0] = epoch + 1
                optuna_score_array[epoch - pre_epoch, 1] = optuna_latest_score
                if trial.should_prune(epoch + 1):
                    logger.info("epoch = {:>5},    loss = {:>10.6f},    [Pruned]".format(epoch + 1, score))
                    # noinspection PyTypeChecker
                    np.savetxt(config.result_directory + os.sep + config.optuna_score_file.full_name,
                               # optuna_score_array[: epoch - pre_epoch], fmt="%.18e", delimiter=",")
                               optuna_score_array[: epoch - pre_epoch], fmt=["%.0f", "%.18e"], delimiter=",")
                    raise TrialPruned

        if trial is not None:
            # noinspection PyTypeChecker
            np.savetxt(config.result_directory + os.sep + config.optuna_score_file.full_name, optuna_score_array,
                       # fmt="%.18e", delimiter=",")
                       fmt=["%.0f", "%.18e"], delimiter=",")

        score = cls.calculate_score(train_keys, validation_keys, result_dict, logger=logger)
        if score < latest_score:
            latest_score = score
            torch.save(model_set.model.state_dict(), best_model_path)
            logger.info("モデル保存: path = {}".format(best_model_path))

        return TrainLog(tuple(data_loader_dict.keys()), loss_array, score_array, latest_score)

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

        debugging: bool = is_debugging()

        for data in data_loader:

            data_length: int = get_data_length(data)
            data_count += data_length

            def _train_for_each_iteration() -> TrainResult:
                if backpropagate:
                    model_set.optimizer.zero_grad()
                    result = cls.train_for_each_iteration(
                        model_set=model_set, data=data, backpropagate=backpropagate, logger=logger, **kwargs)
                    model_set.optimizer.step()
                else:
                    with torch.no_grad():
                        result = cls.train_for_each_iteration(
                            model_set=model_set, data=data, backpropagate=backpropagate, logger=logger, **kwargs)
                return result

            if debugging:
                with torch.autograd.detect_anomaly():
                    train_result: TrainResult = _train_for_each_iteration()
            else:
                train_result = _train_for_each_iteration()

            if issubclass(type(train_result.loss), int) or issubclass(type(train_result.loss), float):
                loss_sum += train_result.loss * data_length
            else:
                raise NotImplementedError

            if is_output_progress:
                score: Optional[ScoreLike] = cls.score_for_each_iteration(
                        model_set=model_set, data=data, train_result=train_result, logger=logger, **kwargs)
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

        base_result_directory: str = config.result_directory
        base_interim_directory: str = config.interim_directory

        def objective(trial: Optional[Trial] = None) -> float:

            if trial is not None:
                util_logger.info("Trial {:d}".format(trial.number + 1))

            # noinspection PyUnusedLocal
            model_set: Optional[ModelSet] = None

            if trial is None or trial.number >= config.train.optuna_pre_trials:
                if config.train.pre_epoch is None:
                    model_set = create_model_set(config=config, device=device, trial=trial, logger=util_logger)
                else:
                    model_set = load_interim_model_set(config=config, device=device, trial=trial, logger=util_logger)

            if trial is not None:
                config.interim_directory = base_interim_directory + os.sep + str(trial.number + 1)
                config.result_directory = base_result_directory + os.sep + str(trial.number + 1)

            os.makedirs(config.interim_directory, exist_ok=True)
            os.makedirs(config.result_directory, exist_ok=True)

            try:
                if trial is None or trial.number >= config.train.optuna_pre_trials:
                    train_log: TrainLog = cls.train(
                        config=config, model_set=model_set,
                        train_dataset=train_dataset, validation_dataset=validation_dataset,
                        visualizes_loss_on_logscale=visualizes_loss_on_logscale,
                        trial=trial, logger=util_logger, **kwargs)
                else:
                    with open(config.result_directory + os.sep + config.optuna_params_file.full_name, "r") as f:
                        actual_params: Dict[str, Any] = json5.load(f)
                    for origin_params in [config.model_set.model.parameters,
                                          config.model_set.criterion.parameters,
                                          config.model_set.optimizer.parameters]:
                        for key, param in origin_params.items():
                            if issubclass(type(param), OptunaParameter):
                                OptunaParameter(
                                    param.suggestion_type, param.name, actual_params[key], actual_params[key]
                                ).get_optuna_parameter(trial)

                    optuna_score_array: np.ndarray = np.loadtxt(
                        config.result_directory + os.sep + config.optuna_score_file.full_name, delimiter=",")
                    for i in range(optuna_score_array.shape[0]):
                        epoch_i: int = round(optuna_score_array[i, 0])
                        trial.report(optuna_score_array[i, 1], epoch_i)
                        if trial.should_prune(epoch_i):
                            raise TrialPruned
                    return optuna_score_array[-1, 1]
            except TrialPruned as e:
                util_logger.info("訓練が途中で打ち切られました。")
                raise e
            except Exception as e:
                util_logger.warning("エラーが発生しました。")
                util_logger.warning(str(e.args))
                raise e

            # model_set.model.eval()

            util_logger.debug("結果を保存します。(ディレクトリ = {})".format(config.result_directory))

            # model.pth
            shutil.copy(config.interim_directory + os.sep + config.model_file.full_name,
                        config.result_directory + os.sep + config.model_file.full_name)
            # torch.save(model_set.model.state_dict(), config.result_directory + os.sep + config.model_file.full_name)

            # loss.csv, score.csv
            fmt_base: List[str] = ["%.0f"]
            fmt_base.extend(["%.18e"] * (train_log.loss_array.shape[1] - 1))
            # noinspection PyTypeChecker
            np.savetxt(config.result_directory + os.sep + config.loss_file.full_name, train_log.loss_array,
                       fmt=fmt_base, delimiter=",")
            if np.any(train_log.score_array != 0):
                # noinspection PyTypeChecker
                np.savetxt(config.result_directory + os.sep + config.score_file.full_name, train_log.score_array,
                           fmt=fmt_base, delimiter=",")

            # params.json5
            if trial is not None:
                with open(config.result_directory + os.sep + config.optuna_params_file.full_name, "w") as f:
                    json5.dump(trial.params, f)

            # loss.png
            visualize_loss(train_log.loss_array, train_log.data_keys, directory=config.result_directory,
                           pre_epoch=config.train.pre_epoch, is_logscale=visualizes_loss_on_logscale)

            cls.run_append(config=config, model_set=model_set, result_directory=config.result_directory,
                           dataset=train_dataset, validation_dataset=validation_dataset, train_log=train_log,
                           device=device, logger=logger)

            config.interim_directory = base_interim_directory
            config.result_directory = base_result_directory

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

            best_result_directory: str = base_result_directory + os.sep + str(study.best_trial.number + 1)
            [shutil.copy(best_result_directory + os.sep + file_name, base_result_directory) for file_name in [
                config.model_file.full_name,
                config.loss_file.full_name,
                config.loss_file.name_without_ext + os.extsep + "png",
                config.score_file.full_name,
                config.score_file.name_without_ext + os.extsep + "png",
                config.optuna_params_file.full_name,
            ]]

        else:
            objective()

        util_logger.close()

    @classmethod
    def run_append(cls, config: ConfigBase, model_set: ModelSet, result_directory: str,
                   dataset: DatasetLike, validation_dataset: Optional[DatasetLike], train_log: TrainLog,
                   *, device: Optional[Device] = None, logger: Optional[UtilLogger] = None, **kwargs) -> None:
        pass


def calculate_loss_sum(keys: Tuple[str, ...], epoch_result_dict: Dict[str, EpochResult]) -> float:
    data_sum: int = sum(result.data_count for key, result in epoch_result_dict.items()
                        if key in keys)
    loss_sum: float = sum(result.loss * result.data_count for key, result in epoch_result_dict.items()
                          if key in keys)
    return loss_sum / data_sum if data_sum > 0 else 0


def calculate_score_sum(keys: Tuple[str, ...], epoch_result_dict: Dict[str, EpochResult]) -> Optional[float]:
    data_sum: int = sum(result.data_count for key, result in epoch_result_dict.items()
                        if ((key in keys) and (result.score is not None)))
    score_sum: float = sum(result.score * result.data_count for key, result in epoch_result_dict.items()
                           if ((key in keys) and (result.score is not None)))
    return score_sum / data_sum if data_sum > 0 else None
