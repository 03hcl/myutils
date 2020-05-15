from ..exceptions import UtilsException


class TorchHelperException(UtilsException):
    def __init__(self, message: str = None, default_message: str = "torch helper error.", *args, **kwargs):
        super(UtilsException, self).__init__(message, default_message, *args, **kwargs)


class CouldNotSuggestOptunaParameterError(TorchHelperException):
    pass
