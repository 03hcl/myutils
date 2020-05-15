import re
from typing import Dict, Optional

from ...const_singleton import ConstSingleton

from .interval import Interval
from .quality import Quality
from .exceptions import NotAnIntervalError


class _Data(ConstSingleton):

    PrefixDict: Dict[str, Quality] = {
        "diminished": Quality.Diminished,
        "diminish": Quality.Diminished,
        "dim": Quality.Diminished,
        "minor": Quality.Minor,
        "min": Quality.Minor,
        "Perfect": Quality.Perfect,
        "perf": Quality.Perfect,
        "major": Quality.Major,
        "maj": Quality.Major,
        "augmented": Quality.Augmented,
        "augment": Quality.Augmented,
        "aug": Quality.Augmented,
    }

    IntervalDict: Dict[Quality, Dict[int, Interval]] = {
        Quality.Diminished: {
            2: Interval.d2,
            3: Interval.d3,
            4: Interval.d4,
            5: Interval.d5,
            6: Interval.d6,
            7: Interval.d7,
            8: Interval.d8,
            9: Interval.d9,
            10: Interval.d10,
            11: Interval.d11,
            12: Interval.d12,
            13: Interval.d13,
            14: Interval.d14,
            15: Interval.d15,
        },
        Quality.Minor: {
            2: Interval.m2,
            3: Interval.m3,
            6: Interval.m6,
            7: Interval.m7,
            9: Interval.m9,
            10: Interval.m10,
            13: Interval.m13,
            14: Interval.m14,
        },
        Quality.Perfect: {
            1: Interval.P1,
            4: Interval.P4,
            5: Interval.P5,
            8: Interval.P8,
            11: Interval.P11,
            12: Interval.P12,
            15: Interval.P15,
        },
        Quality.Major: {
            2: Interval.M2,
            3: Interval.M3,
            6: Interval.M6,
            7: Interval.M7,
            9: Interval.M9,
            10: Interval.M10,
            13: Interval.M13,
            14: Interval.M14,
        },
        Quality.Augmented: {
            1: Interval.A1,
            2: Interval.A2,
            3: Interval.A3,
            4: Interval.A4,
            5: Interval.A5,
            6: Interval.A6,
            7: Interval.A7,
            8: Interval.A8,
            9: Interval.A9,
            10: Interval.A10,
            11: Interval.A11,
            12: Interval.A12,
            13: Interval.A13,
            14: Interval.A14,
            15: Interval.A15,
        },
    }


DATA: _Data = _Data.get_instance()


class Interpreter:

    @classmethod
    def to_interval(cls, value: str) -> Interval:

        match = re.match(r"^\s*([^\d]*?)\s*(\d+?)\s*$", value)
        if not match:
            raise NotAnIntervalError

        prefix: str = match.group(1)
        # noinspection PyUnusedLocal
        q: Quality
        try:
            q = Quality(prefix)
        except ValueError:
            if prefix.lower() in DATA.PrefixDict.keys():
                q = DATA.PrefixDict[prefix.lower()]
            else:
                raise NotAnIntervalError

        number: int = int(match.group(2))
        if number in DATA.IntervalDict[q].keys():
            return DATA.IntervalDict[q][number]

        raise NotAnIntervalError

    @classmethod
    def to_interval_mirex(cls, value: str) -> Interval:

        match = re.match(r"^\s*([^\d]*?)?\s*(\d+?)\s*$", value)
        if not match:
            raise NotAnIntervalError

        prefix: str = match.group(1).strip()
        number: int = int(match.group(2))

        # noinspection PyUnusedLocal
        interval: Optional[Interval]

        if prefix == "b" or prefix == "-":
            interval = DATA.IntervalDict[Quality.Minor].get(number, None) \
                       or DATA.IntervalDict[Quality.Diminished].get(number, None)
        elif prefix == "#" or prefix == "+":
            interval = DATA.IntervalDict[Quality.Augmented].get(number, None)
        else:
            interval = DATA.IntervalDict[Quality.Major].get(number, None) \
                       or DATA.IntervalDict[Quality.Perfect].get(number, None)

        if interval:
            return interval

        raise NotAnIntervalError

