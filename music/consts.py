from ..const_singleton import ConstSingleton


class Consts(ConstSingleton):
    TET: int = 12
    CENT: int = 100
    DEFAULT_NOTE_NUMBER: int = 69
    DEFAULT_STANDARD_PITCH: int = 440


consts = Consts.get_instance()

# TET: int = 12
# CENT: int = 100
# DEFAULT_NOTE_NUMBER: int = 69
# DEFAULT_STANDARD_PITCH: int = 440
