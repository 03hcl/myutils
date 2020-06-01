import inspect


def is_debugging() -> bool:
    for frame in inspect.stack():
        if frame[1].endswith("pydevd.py"):
            return True
    return False
