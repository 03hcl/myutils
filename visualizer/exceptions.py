from ..exceptions import UtilsException


class VisualizerException(UtilsException):
    def __init__(self, message: str = None, default_message: str = "visualizer error.", *args, **kwargs):
        super(VisualizerException, self).__init__(message, default_message, *args, **kwargs)


class InvalidNumberOfDigitsAfterDecimalPointError(VisualizerException):
    pass
