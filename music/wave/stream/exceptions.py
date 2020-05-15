from ...exceptions import MusicException


class WaveIOException(MusicException):
    def __init__(self, message: str = None, default_message: str = "wave io error.", *args, **kwargs):
        super(WaveIOException, self).__init__(message, default_message, *args, **kwargs)


class IncorrectWaveFormatError(WaveIOException):
    pass


class UnreadDataOffsetError(WaveIOException):
    pass


class NotFoundDataChunkError(WaveIOException):
    pass


class UnexpectedStreamPositionError(WaveIOException):
    pass
