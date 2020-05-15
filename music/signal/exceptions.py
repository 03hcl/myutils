from ..exceptions import MusicException


class SignalException(MusicException):
    def __init__(self, message: str = None, default_message: str = "signal error.", *args, **kwargs):
        super(SignalException, self).__init__(message, default_message, *args, **kwargs)


class NotExistKeyTupleError(SignalException):
    pass


class NotMatchNumberOfChannelsError(SignalException):
    pass


class WindowSizeError(SignalException):
    pass
