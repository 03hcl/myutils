from enum import Enum, auto
import math
from typing import Iterable, List, Set, Tuple, TypeVar, Union

import numpy as np
from numpy import log2, log10

from ..consts import consts
from ..note import Note

from .exceptions import NotAPitchError

_pitch = TypeVar("_pitch", bound="Pitch")


class Pitch:

    class Notation(Enum):

        Unknown = 0

        Scientific = auto()
        International = Scientific
        SPN = Scientific
        IPN = Scientific
        SONAR = Scientific
        Cakewalk = SONAR

        YAMAHA = auto()
        Cubase = YAMAHA

        FLStudio = auto()

    # region Property

    @property
    def note_number(self) -> int:
        return self._note_number

    @property
    def cent(self) -> float:
        return self._cent

    @property
    def standard_pitch(self) -> float:
        return self._standard_pitch

    @property
    def frequency(self) -> float:
        return self._frequency

    @property
    def note(self) -> Note:
        return self._note

    @property
    def octave(self) -> int:
        return self._octave

    # endregion

    def __init__(self, note_number: int,
                 *, cent: float = 0, standard_pitch: float = consts.DEFAULT_STANDARD_PITCH, **kwargs):

        cent_q, cent_mod = divmod(cent + consts.CENT / 2, consts.CENT)
        self._note_number: int = note_number + int(cent_q)
        self._cent: float = cent_mod - consts.CENT / 2
        self._standard_pitch: float = Pitch._get_standard_pitch(standard_pitch)

        self._frequency: float = self.standard_pitch * (
                2 ** ((self._note_number - consts.DEFAULT_NOTE_NUMBER + self._cent / consts.CENT) / consts.TET))

        self._note: Note = Note(self._note_number % consts.TET)
        self._octave: int = math.ceil((self._note_number + 1) / consts.TET) - 2

        super(Pitch, self).__init__(**kwargs)

    # region Special Method

    def __str__(self) -> str:
        return "{} [{: >21.15f} Hz] / base: {} Hz".format(
            self.show_note_details(is_shortened=False), self._frequency, self._standard_pitch)

    # endregion

    # region Method

    def cent_from_note_number(self, note_number: int) -> float:
        return self.cent + (self.note_number - note_number) * consts.CENT

    def show_note_details(self, *,
                          is_shortened: bool = True,
                          notation: Notation = Notation.Scientific) -> str:
        notated_octave: int = Pitch._to_notated_octave(self._octave, notation)
        value: str = str(self._note) + str(notated_octave)
        if is_shortened and self._is_zero_cent():
            return value
        return "{:<4} ({})".format(value, self.show_cent(is_shortened=is_shortened))

    def show_cent(self, *, is_shortened: bool = True) -> str:
        if is_shortened:
            if self._is_zero_cent():
                return "0 cents"
            digits: int = 0
            # noinspection PyUnusedLocal
            width: int = 0
            if abs(self._cent) < 1:
                digits = int(-log10(abs(self._cent)) + 1)
                width = digits + 4
            else:
                width: int = int(log10(round(abs(self._cent)))) + 3
            return ("{: =+" + str(width) + "." + str(digits) + "f} cents").format(self._cent)
        return "{: =+20.15f} cents".format(self._cent)

    # endregion

    # region Private Method

    def _is_zero_cent(self, digits: int = 10):
        return round(self._cent, digits) == 0

    # endregion

    # region Class Method

    @classmethod
    def from_note_and_octave(cls, note: Note, notated_octave: int,
                             *, cent: float = 0, standard_pitch: float = consts.DEFAULT_STANDARD_PITCH,
                             notation: Notation = Notation.Scientific):
        octave: int = Pitch._to_octave(notated_octave, notation)
        note_number: int = int(note) + (octave + 1) * consts.TET
        return Pitch(note_number, cent=cent, standard_pitch=standard_pitch)

    @classmethod
    def from_frequency(cls, value: Union[int, float, Iterable[int], Iterable[float], np.ndarray],
                       *, standard_pitch: float = consts.DEFAULT_STANDARD_PITCH) \
            -> Union[_pitch, Tuple[_pitch, ...], Set[_pitch], List[_pitch], np.ndarray]:
        if type(value) == int or type(value) == float:
            return Pitch._create_from_frequency(float(value), standard_pitch=standard_pitch)
        result: List[_pitch] = \
            [Pitch._create_from_frequency(float(freq), standard_pitch=standard_pitch) for freq in value]
        if type(value) == tuple:
            return tuple(result)
        if type(value) == set:
            return set(result)
        if type(value) == list or type(value) == Iterable:
            return result
        if type(value) == np.ndarray:
            return np.array(result, dtype=Pitch)
        raise NotImplementedError

    # endregion

    # region Private Class Method

    @classmethod
    def _create_from_frequency(cls, value: float, *, standard_pitch: float = consts.DEFAULT_STANDARD_PITCH) -> _pitch:
        if math.isinf(value) or math.isnan(value):
            raise NotAPitchError
        cent_mod, cent_q = math.modf(log2(value / standard_pitch) * consts.TET + 0.5)
        note_number: int = consts.DEFAULT_NOTE_NUMBER + int(cent_q)
        cent: float = (cent_mod - 0.5) * consts.CENT
        return cls(note_number, cent=cent, standard_pitch=standard_pitch)

    @classmethod
    def _get_standard_pitch(cls, value: float) -> float:
        return value if value > 0 else consts.DEFAULT_STANDARD_PITCH

    @classmethod
    def _to_octave(cls, notated_octave: int, notation: Notation) -> int:
        if notation == Pitch.Notation.Scientific:
            return notated_octave
        if notation == Pitch.Notation.YAMAHA:
            return notated_octave + 1
        if notation == Pitch.Notation.FLStudio:
            return notated_octave - 1
        raise NotImplementedError

    @classmethod
    def _to_notated_octave(cls, octave: int, notation: Notation) -> int:
        if notation == Pitch.Notation.Scientific:
            return octave
        if notation == Pitch.Notation.YAMAHA:
            return octave - 1
        if notation == Pitch.Notation.FLStudio:
            return octave + 1
        raise NotImplementedError

    # endregion
