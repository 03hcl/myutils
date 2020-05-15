from ..exceptions import MusicException


class ChordException(MusicException):
    def __init__(self, message: str = None, default_message: str = "chord error.", *args, **kwargs):
        super(ChordException, self).__init__(message, default_message, *args, **kwargs)


class NotFoundRootNoteError(ChordException):
    pass


class CollideNoteError(ChordException):
    pass


class CannotInterpretRootError(ChordException):
    pass


class CannotInterpretNoteError(ChordException):
    def __init__(self, key: int, *args, **kwargs):
        super(CannotInterpretNoteError, self).__init__(message=str(key), *args, **kwargs)


class CannotDecodeChordError(ChordException):
    pass


class CannotEncodeWholeChordStringError(ChordException):
    def __init__(self, rest_str: str, *args, **kwargs):
        super(CannotEncodeWholeChordStringError, self).__init__(message=rest_str, *args, **kwargs)


class CannotEncodeWholeTensionStringError(ChordException):
    def __init__(self, rest_str: str, *args, **kwargs):
        super(CannotEncodeWholeTensionStringError, self).__init__(message=rest_str, *args, **kwargs)
