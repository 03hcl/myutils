class SourceException(Exception):

    @property
    def message(self):
        return self._message

    def __init__(self, message: str = None, default_message: str = "my source exception.", *args, **kwargs):
        super(SourceException, self).__init__(message, args, kwargs)
        self._message: str = default_message if message is None else message
