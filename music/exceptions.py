from ..exception_base import SourceException


class MusicException(SourceException):
    def __init__(self, message: str = None, default_message: str = "music error.", *args, **kwargs):
        super(MusicException, self).__init__(message, default_message, *args, **kwargs)
