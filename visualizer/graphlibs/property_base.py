from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from .exceptions import CannotUpdatePropertyValueError

T = TypeVar("T")


class PropertyBase(ABC, Generic[T]):

    def __init__(self, *args, alt: Optional[T] = None, **kwargs):
        self._value: T = None
        super(PropertyBase, self).__init__(*args, **kwargs)
        self.update(alt)

    def __call__(self, value: Optional[T] = None, **kwargs) -> T:
        if value is not None:
            self._set(value, **kwargs)
            self.update(alt=value, **kwargs)
        return self._value

    def update(self, alt: Optional[T] = None, **kwargs) -> None:
        value: Optional[T] = self._get(**kwargs)
        if value is None:
            if alt is None:
                raise CannotUpdatePropertyValueError
            self._value = alt
        else:
            self._value = value

    @abstractmethod
    def _get(self, **kwargs) -> Optional[T]:
        pass

    @abstractmethod
    def _set(self, value: T, **kwargs) -> None:
        pass
