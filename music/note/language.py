from enum import auto, Enum, unique
from typing import TypeVar

_note_language = TypeVar("_note_language", bound="NoteLanguage")


@unique
class Language(Enum):

    # region Enum

    Unknown = 0x0

    MIDI = 0x1
    English = auto()
    Italian = auto()
    German = auto()
    Japanese = auto()

    Dutch = auto()
    Byzantine = auto()
    Indian = auto()

    ItalianInKana = 0x1000
    JapaneseInRoman = auto()

    Chromatic = 0x10000
    FixedDoInRoman = auto()
    FixedDoInKana = auto()

    # endregion

    @classmethod
    def from_str(cls, value: str) -> _note_language:
        from .interpreter import Interpreter
        return Interpreter.get_language_from_str(value)


