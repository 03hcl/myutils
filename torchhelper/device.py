from typing import Optional

import torch

from .. import UtilLogger


class Device:

    def __init__(self, *, gpu_number: int = -1, gpu_name: str = "", logger: Optional[UtilLogger] = None):

        if logger is not None:
            logger.info("torch 用の Device を作成します。")
        if gpu_number < 0:
            if gpu_name:
                gpu_number = Device._search_gpu_number(gpu_name)
                if gpu_number < 0:
                    if logger is not None:
                        logger.warning("デバイス名が一致する CUDA デバイスは存在しなかったため CPU を使用します。")
            else:
                # logger_wrapper.info("CPU を使用します。")
                pass
        else:
            if torch.cuda.is_available():
                if gpu_name:
                    if gpu_number >= torch.cuda.device_count() or \
                            gpu_name.replace(" ", "").lower() not in \
                            torch.cuda.get_device_name(gpu_number).replace(" ", "").lower():
                        if logger is not None:
                            logger.warning("デバイス番号とデバイス名が一致しないため、デバイス名を検索します。")
                        n: int = Device._search_gpu_number(gpu_name)
                        if n < 0:
                            if logger is not None:
                                logger.warning("デバイス番号とデバイス名が一致する CUDA デバイスは存在しなかったため、指定したデバイス番号の GPU を使用します。")
                        else:
                            gpu_number = n
                if gpu_number >= torch.cuda.device_count():
                    if logger is not None:
                        logger.warning("指定したデバイス番号の CUDA デバイスは存在しないため CPU を使用します。")
                    gpu_number = -1
            else:
                if logger is not None:
                    logger.warning("システムが CUDA をサポートしていないため CPU を使用します。")
                gpu_number = -1

        self.gpu_number: int = gpu_number
        self.gpu_name: str = ""
        self.torch_device_name: str = ""

        self.is_cpu: bool = self.gpu_number < 0
        self.is_gpu: bool = self.gpu_number >= 0

        if self.is_gpu:
            self.gpu_name = torch.cuda.get_device_name(self.gpu_number)
            self.torch_device_name = "cuda:{:d}".format(self.gpu_number)
            if logger is not None:
                logger.info("GPU を使用します。")
                logger.info("デバイス番号: {:d}".format(self.gpu_number))
                logger.info("デバイス名: {}".format(self.gpu_name))
        if self.is_cpu:
            self.torch_device_name = "cpu"
            if logger is not None:
                logger.info("CPU を使用します。")

        self.torch_device: torch.device = torch.device(self.torch_device_name)

    @staticmethod
    def _search_gpu_number(gpu_name: str) -> int:
        gpu_name = gpu_name.replace(" ", "").lower()
        for i in range(torch.cuda.device_count()):
            if gpu_name in torch.cuda.get_device_name(i).replace(" ", "").lower():
                return 0
        return -1


def adapt_tensor_to_device(tensor: torch.Tensor, device: Device):
    return tensor.to(device.torch_device) if device.is_gpu else tensor
