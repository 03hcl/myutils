from ..exceptions import GraphLibsException


class MatPlotLibException(GraphLibsException):
    def __init__(self, message: str = None, default_message: str = "matplotlib error.", *args, **kwargs):
        super(MatPlotLibException, self).__init__(message, default_message, *args, **kwargs)


class NotFoundAxisScaleStrError(MatPlotLibException):
    pass


class NotFoundAxisScaleFunctionError(MatPlotLibException):
    pass
