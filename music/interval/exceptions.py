from ..exceptions import MusicException


class IntervalException(MusicException):
    def __init__(self, message: str = None, default_message: str = "interval error.", *args, **kwargs):
        super(IntervalException, self).__init__(message, default_message, *args, **kwargs)


class NotAnIntervalError(IntervalException):
    pass


class CannotConvertToIntervalStrError(IntervalException):
    pass
