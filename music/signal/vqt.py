from typing import Union

import numpy as np

from .frequency_domain_signal import FrequencyDomainSignalBase
from .transformer import ConstantQ, VariableQ


class VQT(FrequencyDomainSignalBase):

    @property
    def q(self) -> Union[ConstantQ, VariableQ]:
        return self._q

    def __init__(self, data: np.ndarray, frequencies: np.ndarray, sampling_interval: float, seconds: np.ndarray,
                 calculated_q_transformer: Union[ConstantQ, VariableQ], *args, **kwargs):
        super(VQT, self).__init__(data, frequencies, sampling_interval, seconds, *args, **kwargs)
        self._q: Union[ConstantQ, VariableQ] = calculated_q_transformer
