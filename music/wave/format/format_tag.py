from enum import Enum, unique


@unique
class FormatTag(Enum):

    unknown = 0x0000
    pcm = 0x0001
    adpcm_ms = 0x0002
    ieee_float = 0x0003
    ibm_csvd = 0x0005
    a_law = 0x0006
    mu_law = 0x0007
    ms_gsm = 0x0031
    adpcm_32k = 0x0064
    extensible = 0xFFFE

    def is_linear_pcm(self) -> bool:
        return self == FormatTag.pcm or self == FormatTag.ieee_float
