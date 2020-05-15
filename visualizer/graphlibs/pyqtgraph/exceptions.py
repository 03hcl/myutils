from ..exceptions import GraphLibsException


class PyQtGraphException(GraphLibsException):
    def __init__(self, message: str = None, default_message: str = "pyqtgraph error.", *args, **kwargs):
        super(PyQtGraphException, self).__init__(message, default_message, *args, **kwargs)


