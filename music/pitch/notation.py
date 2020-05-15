from enum import Enum, auto


class Notation(Enum):

    Unknown = 0

    Scientific = auto()
    International = Scientific
    SPN = Scientific
    IPN = Scientific
    SONAR = Scientific
    Cakewalk = SONAR

    YAMAHA = auto()
    Cubase = YAMAHA

    FLStudio = auto()
