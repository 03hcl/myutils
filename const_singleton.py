from typing import Any, Optional, TypeVar

from .exceptions import ConstError

_const_singleton = TypeVar("_const_singleton", bound="ConstSingleton")


class ConstSingleton:

    _unique_instance: Optional[_const_singleton] = None

    def __new__(cls, *args, **kwargs):
        raise ConstError

    def __setattr__(self, name: str, value: Any):
        raise ConstError

    @classmethod
    def get_instance(cls, *args, **kwargs) -> _const_singleton:
        if not cls._unique_instance:
            cls._unique_instance = super(ConstSingleton, cls).__new__(cls, *args, **kwargs)
        return cls._unique_instance
