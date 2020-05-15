from enum import IntEnum, unique
from typing import TypeVar, Union

from ..consts import consts

from .language import Language

_note = TypeVar("_note", bound="Note")


@unique
class Note(IntEnum):

    C = 0
    C_SHARP = 1
    D = 2
    D_SHARP = 3
    E = 4
    F = 5
    F_SHARP = 6
    G = 7
    G_SHARP = 8
    A = 9
    A_SHARP = 10
    B = 11

    def __str__(self) -> str:
        if self == Note.C:
            return "C"
        if self == Note.C_SHARP:
            return "C#"
        if self == Note.D:
            return "D"
        if self == Note.D_SHARP:
            return "D#"
        if self == Note.E:
            return "E"
        if self == Note.F:
            return "F"
        if self == Note.F_SHARP:
            return "F#"
        if self == Note.G:
            return "G"
        if self == Note.G_SHARP:
            return "G#"
        if self == Note.A:
            return "A"
        if self == Note.A_SHARP:
            return "A#"
        if self == Note.B:
            return "B"

    @classmethod
    def from_int(cls, value: int) -> _note:
        return Note(value % consts.TET)

    @classmethod
    def from_str(cls, value: str, language: Union[str, Language] = Language.MIDI) -> _note:
        from .interpreter import Interpreter
        return Interpreter.to_note(value, language)
