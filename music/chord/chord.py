from collections.abc import MutableMapping
from typing import Dict, Iterator, Optional, Set, Tuple, TypeVar, Union, \
    KeysView, ItemsView, ValuesView

from ..consts import consts
from ..interval import Interval
from ..interval.exceptions import NotAnIntervalError
from ..note import Note, NoteLanguage
from ..note.exceptions import NotFoundNoteError

from .notation import Notation
from .exceptions import NotFoundRootNoteError, CannotInterpretNoteError, CollideNoteError

_chord = TypeVar("_chord", bound="Chord")

RootNoteLike = Union[Note, str]
NoteLike = Optional[Union[Note, str, Interval]]

_exists_keys: Tuple[int, ...] = (1, 3, 5, 7, 9, 11, 13)


class Chord(MutableMapping):

    # region Property

    @property
    def notes(self) -> Dict[int, Optional[Note]]:
        return self._notes_dict

    @property
    def intervals(self) -> Dict[int, Optional[Interval]]:
        return self._intervals_dict

    @property
    def root(self) -> Note:
        return self._root

    @property
    def omits_root(self) -> bool:
        return self._omits_root

    @property
    def note_3rd(self) -> Optional[Note]:
        return self._notes_dict[3]

    @property
    def note_5th(self) -> Optional[Note]:
        return self._notes_dict[5]

    @property
    def note_7th(self) -> Optional[Note]:
        return self._notes_dict[7]

    @property
    def note_9th(self) -> Optional[Note]:
        return self._notes_dict[9]

    @property
    def note_11th(self) -> Optional[Note]:
        return self._notes_dict[11]

    @property
    def note_13th(self) -> Optional[Note]:
        return self._notes_dict[13]

    @property
    def interval_3rd(self) -> Optional[Interval]:
        return self._intervals_dict[3]

    @property
    def interval_5th(self) -> Optional[Interval]:
        return self._intervals_dict[5]

    @property
    def interval_7th(self) -> Optional[Interval]:
        return self._intervals_dict[7]

    @property
    def interval_9th(self) -> Optional[Interval]:
        return self._intervals_dict[9]

    @property
    def interval_11th(self) -> Optional[Interval]:
        return self._intervals_dict[11]

    @property
    def interval_13th(self) -> Optional[Interval]:
        return self._intervals_dict[13]

    # endregion

    def __init__(self, root: RootNoteLike, third: NoteLike, fifth: NoteLike,
                 seventh: NoteLike = None, ninth: NoteLike = None,
                 eleventh: NoteLike = None, thirteenth: NoteLike = None,
                 *, omits_root: bool = False, language: Union[str, NoteLanguage] = NoteLanguage.MIDI):

        self._root: Note = root if type(root) == Note else Note.from_str(root, language)
        self._omits_root: bool = omits_root

        self._notes_dict: Dict[int, Optional[Note]] = dict()
        self._intervals_dict: Dict[int, Optional[Interval]] = dict()

        if self._omits_root:
            self._set_none(1)
        else:
            self._notes_dict[1] = self._root
            self._intervals_dict[1] = Interval.P1

        self._interpret_note_and_interval(3, third, language)
        self._interpret_note_and_interval(5, fifth, language)
        self._interpret_note_and_interval(7, seventh, language)
        self._interpret_note_and_interval(9, ninth, language)
        self._interpret_note_and_interval(11, eleventh, language)
        self._interpret_note_and_interval(13, thirteenth, language)

        self._len: int = len([x for x in self._notes_dict.values() if x])

        # self._str: str = "()"
        # if self._len > 0:
        #     # self._str = str(tuple(str(v) for v in self._notes_dict.values() if v is not None))
        #     self._str = "(" + ", ".join(str(v) for v in self._notes_dict.values() if v is not None) + ")"

    # region Special Method

    def __contains__(self, item: Note):
        return item in self._notes_dict.values()

    def __setitem__(self, k: int, v: Optional[Note]) -> None:
        raise NotImplementedError

    def __delitem__(self, v: Note) -> None:
        raise NotImplementedError

    def __getitem__(self, k: int) -> Optional[Note]:
        # if key not in self._exists_intervals:
        #     raise IndexError
        # return self._notes_dict.get(key, None)
        return self._notes_dict[k]

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[Tuple[int, Optional[Note]]]:
        # noinspection PyTypeChecker
        return iter(self.items())

    # endregion

    # region Mix-in Mapping Method

    def keys(self) -> KeysView[int]:
        return self._notes_dict.keys()

    def items(self) -> ItemsView[int, Optional[Note]]:
        return self._notes_dict.items()

    def values(self) -> ValuesView[Optional[Note]]:
        return self._notes_dict.values()

    def get(self, k: int) -> Optional[Note]:
        return self._notes_dict.get(k, None)

    def __eq__(self, other: Union[_chord, Dict[int, Optional[Union[Note, Interval]]]]) -> bool:

        if type(other) == Chord:
            # noinspection PyProtectedMember
            return self._notes_dict == other._notes_dict

        elif type(other) == dict:

            # noinspection PyUnusedLocal
            k: int

            for k in _exists_keys:

                v_other: Optional[Union[Note, Interval]] = other.get(k, None)

                # noinspection PyUnusedLocal
                v_self: Optional[Union[Note, Interval]]
                if type(v_other) == Note:
                    v_self = self._notes_dict.get(k, None)
                else:
                    v_self = self._intervals_dict.get(k, None)

                if v_self != v_other:
                    return False

            return True

        raise TypeError

    def __ne__(self, other: Union[_chord, Dict[int, Optional[Note]]]) -> bool:
        return not self == other

    # endregion

    # region Mix-in MutableMapping Method (Not Implemented)

    def pop(self, k: int) -> Note:
        raise NotImplementedError

    def popitem(self) -> Tuple[int, Note]:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def update(self, __m, **kwargs) -> None:
        raise NotImplementedError

    def setdefault(self, k: int, default: Note = ...) -> Note:
        raise NotImplementedError

    # endregion

    def __str__(self) -> str:
        # return str(tuple(str(v) for v in self._notes_dict.values() if v is not None))
        # return self._str
        return "(" + ", ".join(str(v) for v in self._notes_dict.values() if v is not None) + ")" \
            if self._len > 0 else "()"

    def encode(self, notation: Notation = Notation.Standard):
        from .interpreter import Interpreter
        return Interpreter.encode_to_str(self, notation)

    # region Private Method

    def _interpret_note_and_interval(self, key: int, value: NoteLike,
                                     language: Union[str, NoteLanguage] = NoteLanguage.MIDI) -> None:
        if value is None:
            self._set_none(key)
        elif type(value) == Note:
            self._set_from_note(key, value)
        elif type(value) == Interval:
            self._set_from_interval(key, value)
        elif type(value) == str:
            if str.isspace(value):
                self._set_none(key)
            else:
                try:
                    self._set_from_note(key, Note.from_str(value, language))
                except NotFoundNoteError:
                    try:
                        self._set_from_interval(key, Interval.from_str(value))
                    except NotAnIntervalError:
                        raise CannotInterpretNoteError
        else:
            raise CannotInterpretNoteError

    def _set_none(self, key: int):
        self._notes_dict[key] = None
        self._intervals_dict[key] = None

    def _set_from_interval(self, key: int, value: Interval):
        self._notes_dict[key] = Note.from_int(self.root.value + value.value)
        self._intervals_dict[key] = value

    def _set_from_note(self, key: int, value: Note):
        self._notes_dict[key] = value
        self._intervals_dict[key] = (int(value) - int(self._root)) % consts.TET + (key - 1) // 7 * consts.TET

    # endregion

    # region Class Method

    @classmethod
    def from_dict(cls, notes_dict: Dict[int, NoteLike],
                  *, root_note: Optional[RootNoteLike] = None,
                  omits_root: bool = False, language: Union[str, NoteLanguage] = NoteLanguage.MIDI) -> _chord:

        root: NoteLike = notes_dict.get(1, None) if root_note is None else root_note
        if root is None or type(root) == Interval:
            raise NotFoundRootNoteError

        return Chord(root, notes_dict.get(3, None), notes_dict.get(5, None),
                     notes_dict.get(7, None), notes_dict.get(9, None),
                     notes_dict.get(11, None), notes_dict.get(13, None),
                     omits_root=omits_root, language=language)

    @classmethod
    def compose(cls, base_chord: _chord, composed: Dict[int, NoteLike],
                *, language: Union[str, NoteLanguage] = NoteLanguage.MIDI, prior_interval: Optional[Set[int]] = None) \
            -> _chord:

        base: Chord = base_chord
        prior_interval = prior_interval or {}

        composed_root: NoteLike = composed.get(1, None)
        if type(composed_root) == Interval and composed_root != Interval.P1:
            raise CollideNoteError
        composed_root_note: Optional[Note] = None
        if type(composed_root) == str and not str.isspace(composed_root):
            composed_root_note = Note.from_str(composed_root, language)
        elif type(composed_root) == Note:
            composed_root_note = composed_root
        if composed_root_note is not None and base.root != composed_root_note:
            raise CollideNoteError

        # noinspection PyUnusedLocal
        k: int
        # noinspection PyUnusedLocal
        v: Optional[int]

        for k, v in base.items():
            composed_value: NoteLike = composed.get(k, None)
            if k in prior_interval:
                if k not in composed:
                    composed[k] = v
            else:
                if composed_value is not None:
                    raise CollideNoteError
                composed[k] = v

        return Chord.from_dict(composed, root_note=base.root, omits_root=base.omits_root, language=language)

    @classmethod
    def from_str(cls, value: str, notation: Notation = Notation.Standard, language: NoteLanguage = NoteLanguage.MIDI) \
            -> _chord:
        from .interpreter import Interpreter
        return Interpreter.decode_from_str(value, notation, language, requires_bass=False)

    # endregion


NoneChord = Chord(Note.C, None, None, omits_root=True)
