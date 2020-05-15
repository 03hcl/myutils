from .exception_base import SourceException


class UtilsException(SourceException):
    def __init__(self, message: str = None, default_message: str = "utils error.", *args, **kwargs):
        super(UtilsException, self).__init__(message, default_message, *args, **kwargs)


class CannotInterpretTitleError(UtilsException):
    pass


class ConstError(UtilsException):
    pass


class DuplicateFileTableKeyError(UtilsException):
    pass


class DuplicateModuleArgumentsError(UtilsException):
    pass


class FileNameArgumentsError(UtilsException):
    pass


class NotExistTableItemError(UtilsException):
    pass
