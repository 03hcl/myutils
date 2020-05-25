from typing import Collection

from .channel_mask import ChannelMask
from .format_tag import FormatTag


class WaveFormat:

    # region Property

    @property
    def format_code(self):
        pass

    @property
    def number_of_channels(self):
        pass

    @property
    def sampling_rate(self):
        pass

    @property
    def data_rate(self):
        pass

    @property
    def data_block_size(self):
        pass

    @property
    def bits_per_sample(self):
        pass

    @property
    def bytes_per_sample(self):
        pass

    @property
    def number_of_valid_bits(self):
        pass

    @property
    def speaker_position_mask(self):
        pass

    @property
    def sub_format(self):
        pass

    @property
    def number_of_samples(self):
        pass

    # endregion

    # region Getter, Setter

    # noinspection PyUnresolvedReferences
    @format_code.getter
    def format_code(self) -> FormatTag:
        return self._w_format_tag

    @format_code.setter
    def format_code(self, value: FormatTag):
        self._w_format_tag = value

    @number_of_channels.getter
    def number_of_channels(self) -> int:
        return self._n_channels

    @number_of_channels.setter
    def number_of_channels(self, value: int):
        self._n_channels = value

    @sampling_rate.getter
    def sampling_rate(self) -> int:
        return self._n_samples_per_sec

    @sampling_rate.setter
    def sampling_rate(self, value: int):
        self._n_samples_per_sec = value

    @data_rate.getter
    def data_rate(self) -> int:
        return self._n_avg_bytes_per_sec

    @data_rate.setter
    def data_rate(self, value: int):
        self._n_avg_bytes_per_sec = value

    @data_block_size.getter
    def data_block_size(self) -> int:
        return self._n_block_align

    @data_block_size.setter
    def data_block_size(self, value: int):
        self._n_block_align = value

    @bits_per_sample.getter
    def bits_per_sample(self) -> int:
        return self._w_bits_per_sample

    @bytes_per_sample.getter
    def bytes_per_sample(self) -> int:
        return self.bits_per_sample // 8

    @bits_per_sample.setter
    def bits_per_sample(self, value: int):
        self._w_bits_per_sample = value

    @number_of_valid_bits.getter
    def number_of_valid_bits(self) -> int:
        return self._w_valid_bits_per_sample

    @number_of_valid_bits.setter
    def number_of_valid_bits(self, value: int):
        self._w_valid_bits_per_sample = value

    @speaker_position_mask.getter
    def speaker_position_mask(self) -> ChannelMask:
        return self._dw_channel_mask

    @speaker_position_mask.setter
    def speaker_position_mask(self, value: ChannelMask):
        self._dw_channel_mask = value

    @sub_format.getter
    def sub_format(self) -> bytes:
        return self._sub_format

    @sub_format.setter
    def sub_format(self, value: bytes):
        self._sub_format = value

    @number_of_samples.getter
    def number_of_samples(self) -> Collection[int]:
        return self._dw_sample_length

    # endregion

    def __init__(self, *,
                 format_code: FormatTag = FormatTag.unknown,
                 n_channels: int = 0,
                 sampling_rate: int = 0,
                 sample_size: int = 0,
                 valid_size: int = 0,
                 channel_mask: ChannelMask = ChannelMask.unknown,
                 sub_format: bytes = b"",
                 n_samples: Collection[int] = (),
                 **kwargs):
        self._n_channels: int = n_channels
        self._w_format_tag: FormatTag = format_code
        self._n_samples_per_sec: int = sampling_rate
        self._n_avg_bytes_per_sec: int = sample_size * n_channels * sampling_rate
        self._n_block_align: int = sample_size * n_channels
        self._w_bits_per_sample: int = sample_size * 8
        self._w_valid_bits_per_sample: int = valid_size * 8
        self._dw_channel_mask: ChannelMask = channel_mask
        self._sub_format: bytes = sub_format
        self._dw_sample_length: Collection[int] = list(n_samples)
        super(WaveFormat, self).__init__(**kwargs)

    @classmethod
    def create_default(cls, *, n_channels: int, sampling_rate: int):
        return WaveFormat(format_code=FormatTag.ieee_float,
                          n_channels=n_channels,
                          sampling_rate=sampling_rate,
                          sample_size=4)
