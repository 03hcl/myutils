import datetime
from logging import Logger, getLogger, StreamHandler, FileHandler, Formatter, DEBUG
import os
from typing import Optional

import numpy as np
from optuna.trial import Trial
import torch
import torch.nn as nn

from .config_base import ConfigBase
from .timemeter import TimeMeter


class UtilLogger:

    def __init__(self, config: Optional[ConfigBase] = None, *, base_logger: Logger = None, **kwargs):
        self.base_logger: Logger = base_logger or UtilLogger.create_base_logger(config, **kwargs)
        self.interim_directory: str = ("." + os.sep + "interim") if config is None else config.interim_directory
        os.makedirs(self.interim_directory, exist_ok=True)
        self.time_meter: TimeMeter = TimeMeter()

        self.debug("UtilLogger を作成しました。次のファイルに記録されます。")
        for handler in self.base_logger.handlers:
            if issubclass(type(handler), FileHandler):
                file_handler: FileHandler = handler
                self.debug(file_handler.baseFilename)

    # region logger-like Method

    def close(self) -> None:
        for h in self.base_logger.handlers:
            self.base_logger.removeHandler(h)

    def debug(self, msg: str = "", *args, **kwargs) -> None:
        self.base_logger.debug(msg, *args, **kwargs)

    def info(self, msg: str = "", *args, **kwargs) -> None:
        self.base_logger.info(msg, *args, **kwargs)

    def warning(self, msg: str = "", *args, **kwargs) -> None:
        self.base_logger.warning(msg, *args, **kwargs)

    def error(self, msg: str = "", *args, **kwargs) -> None:
        self.base_logger.error(msg, *args, **kwargs)

    def critical(self, msg: str = "", *args, **kwargs) -> None:
        self.base_logger.critical(msg, *args, **kwargs)

    # endregion

    def snap_epoch_with_loss(self, loss: float, present_epoch: int, all_epoch: int, *,
                             pre_epoch: int = 0, log_prefix: str = "", log_suffix: str = "",
                             augmented_digit: bool = False,
                             customized_log_str_format: str = "") -> None:

        self.time_meter.snap_present_time(progress=((present_epoch - pre_epoch) / (all_epoch - pre_epoch)))

        log_str_format: str = "epoch = {0:>"
        log_str_format += "10" if augmented_digit and present_epoch >= 100000 else "5"
        log_str_format += "},    loss = "
        if customized_log_str_format:
            log_str_format += customized_log_str_format
        else:
            log_str_format += "{1:>"
            log_str_format += "16.12" if augmented_digit and loss < 1e-5 else "10.6"
            log_str_format += "f}"
        log_str_format += ",    経過: {2} / 残り: {3}"

        log_str: str = ""
        if log_prefix:
            log_str += log_prefix + ", "
        log_str += log_str_format.format(present_epoch, loss,
                                         TimeMeter.format_total_seconds(self.time_meter.elapsed, False),
                                         TimeMeter.format_total_seconds(self.time_meter.estimation, False))
        if log_suffix:
            log_str += ", " + log_suffix

        self.debug(log_str)

    def save_loss_array_and_model(self, config: Optional[ConfigBase], epoch: int,
                                  loss_array: np.ndarray, model: nn.Module,
                                  *, directory: str = None, log_str: str = None, trial: Optional[Trial] = None) -> None:

        if config is None:
            directory = directory or self.interim_directory
            if trial is not None:
                directory += os.sep + str(trial.number + 1)
            os.makedirs(directory, exist_ok=True)
        else:
            directory = config.interim_directory

        epoch_str: str = config.get_epoch_str_function(epoch)
        log_array_path: str = directory + os.sep + config.loss_file.get_full_name(suffix=epoch_str)
        model_path: str = directory + os.sep + config.model_file.get_full_name(suffix=epoch_str)

        # noinspection PyTypeChecker
        np.savetxt(log_array_path, loss_array, fmt="%.18e", delimiter=",")
        torch.save(model.state_dict(), model_path)
        self.debug(log_str or "モデル保存: path = {}".format(model_path))

    @staticmethod
    def create_base_logger(config: Optional[ConfigBase] = None, *,
                           logger_type: Optional[type] = None,
                           directory_path: str = "." + os.sep + "log", file_name: str = "") -> Logger:

        if logger_type is not None:
            return logger_type()

        base_logger = getLogger(__name__)
        base_logger.setLevel(DEBUG)

        stream_handler = StreamHandler()
        stream_handler.setLevel(DEBUG)
        base_logger.addHandler(stream_handler)

        if file_name:
            directory_path = directory_path if config is None else config.log_directory
            os.makedirs(directory_path, exist_ok=True)
            file_path = directory_path + os.sep + "{}_{:%Y%m%d_%H%M%S}{}log".format(
                file_name, datetime.datetime.now(), os.extsep)
            file_handler = FileHandler(filename=file_path,  encoding="utf-8")
            file_handler.setLevel(DEBUG)
            # noinspection SpellCheckingInspection
            file_handler.setFormatter(Formatter("%(asctime)s;    %(message)s"))
            base_logger.addHandler(file_handler)

        base_logger.propagate = False
        return base_logger


class BlankUtilLogger(UtilLogger):

    def __init__(self):
        super(BlankUtilLogger, self).__init__(base_logger=getLogger(__name__))

    def close(self) -> None:
        pass

    def debug(self, msg: str = "", *args, **kwargs) -> None:
        print(msg)

    def info(self, msg: str = "", *args, **kwargs) -> None:
        print(msg)

    def warning(self, msg: str = "", *args, **kwargs) -> None:
        print(msg)

    def error(self, msg: str = "", *args, **kwargs) -> None:
        print(msg)

    def critical(self, msg: str = "", *args, **kwargs) -> None:
        print(msg)
