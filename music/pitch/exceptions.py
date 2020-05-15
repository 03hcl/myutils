from ..exceptions import MusicException


class PitchException(MusicException):
    def __init__(self, message: str = None, default_message: str = "pitch error.", *args, **kwargs):
        super(PitchException, self).__init__(message, default_message, *args, **kwargs)


class NotAPitchError(PitchException):
    pass
