from ..exceptions import SignalException


class TransformerException(SignalException):
    def __init__(self, message: str = None, default_message: str = "transformer error.", *args, **kwargs):
        super(TransformerException, self).__init__(message, default_message, *args, **kwargs)


# class WindowTypeError(TransformerException):
#     pass
