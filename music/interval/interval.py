from enum import IntEnum
from typing import TypeVar

from .quality import Quality
from .exceptions import CannotConvertToIntervalStrError

_interval = TypeVar("_interval", bound="Interval")

# https://en.wikipedia.org/wiki/Interval_(music)


class Interval(IntEnum):

    # region Enum

    P1 = 0
    m2 = 1
    M2 = 2
    m3 = 3
    M3 = 4
    P4 = 5
    d5 = 6
    P5 = 7
    m6 = 8
    M6 = 9
    m7 = 10
    M7 = 11
    P8 = 12
    m9 = 13
    M9 = 14
    m10 = 15
    M10 = 16
    P11 = 17
    d12 = 18
    P12 = 19
    m13 = 20
    M13 = 21
    m14 = 22
    M14 = 23
    P15 = 24

    S = 1
    T = 2
    TT = 6

    # d1 = -1
    A1 = 1

    d2 = 0
    A2 = 3

    d3 = 2
    A3 = 5

    d4 = 4
    A4 = 6

    A5 = 8

    d6 = 7
    A6 = 10

    d7 = 9
    A7 = 12

    d8 = 11
    A8 = 13

    d9 = 12
    A9 = 15

    d10 = 14
    A10 = 17

    d11 = 16
    A11 = 18

    A12 = 20

    d13 = 19
    A13 = 22

    d14 = 21
    A14 = 24

    d15 = 23
    A15 = 25

    # endregion

    def to_str(self, number: int) -> str:
        from .interpreter import DATA
        # noinspection PyUnusedLocal
        q: Quality
        for q in Quality:
            if DATA.IntervalDict[q].get(number) == self:
                return q.value + str(number)
        raise CannotConvertToIntervalStrError

    @classmethod
    def from_str(cls, value: str) -> _interval:
        from .interpreter import Interpreter
        return Interpreter.to_interval(value)
