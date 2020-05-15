from ..exceptions import VisualizerException


class GraphLibsException(VisualizerException):
    def __init__(self, message: str = None, default_message: str = "graphlibs error.", *args, **kwargs):
        super(GraphLibsException, self).__init__(message, default_message, *args, **kwargs)


class CollidePlotLabelError(GraphLibsException):
    pass


class InvalidGridError(GraphLibsException):
    pass


class CannotShowFigureError(GraphLibsException):
    pass


class CannotUpdatePropertyValueError(GraphLibsException):
    pass
