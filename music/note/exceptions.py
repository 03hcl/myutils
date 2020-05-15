from ..exceptions import MusicException


class NoteException(MusicException):
    def __init__(self, message: str = None, default_message: str = "note error.", *args, **kwargs):
        super(NoteException, self).__init__(message, default_message, *args, **kwargs)


class NotFoundNoteLanguageError(NoteException):
    pass


class NotFoundNoteError(NoteException):
    pass


class NotFoundRootNoteError(NoteException):
    pass
