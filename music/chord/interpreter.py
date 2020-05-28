from typing import Dict, Optional, Tuple, TypeVar, Union

from ...const_singleton import ConstSingleton

from ..interval import Interval
from ..note import Note, NoteLanguage
from ..note.exceptions import NotFoundNoteError

from .chord import Chord, NoneChord
from .notation import Notation
from .symbol import Symbol
from .with_bass import ChordWithBass
from .exceptions import \
    CannotDecodeChordError, CannotEncodeWholeChordStringError, CannotEncodeWholeTensionStringError, \
    CannotInterpretRootError, CollideNoteError

T = TypeVar("T")


class _Data(ConstSingleton):

    BaseSuffixDict: Tuple[Symbol, ...] = \
        (
            Symbol("dim", {3: Interval.m3, 5: Interval.d5, 7: Interval.d7},
                   decode={Notation.Standard, Notation.MIREX},
                   encode={Notation.Standard, Notation.MIREX}),
            Symbol("dim", {3: Interval.m3, 5: Interval.d5, 7: Interval.d7}, {"dim7", "0", "○", },
                   decode={Notation.Advanced},
                   encode={Notation.Advanced}),
            Symbol("aug", {3: Interval.M3, 5: Interval.A5, 7: None},
                   decode={Notation.Standard, Notation.MIREX, Notation.Advanced},
                   encode={Notation.Standard, Notation.MIREX}),
            Symbol("aug", {3: Interval.M3, 5: Interval.A5},
                   decode=None, encode={Notation.Advanced}),
            Symbol("hdim7", {3: Interval.m3, 5: Interval.d5, 7: Interval.m7},
                   decode=None, encode={Notation.MIREX}),
            Symbol("∅", {3: Interval.m3, 5: Interval.d5, 7: Interval.m7}, {"Φ", "φ", },
                   decode=None, encode={Notation.Advanced}),

            Symbol("m", {3: Interval.m3}, {"min", "Min", }, excepted={"maj"},
                   decode={Notation.Standard, Notation.MIREX, Notation.Advanced},
                   encode={Notation.Standard, Notation.MIREX, Notation.Advanced}),
            Symbol("5", {3: None, 5: Interval.P5},
                   decode={Notation.Standard, Notation.Advanced},
                   encode={Notation.Standard, Notation.Advanced}),
            Symbol("9", {7: Interval.m7, 9: Interval.M9},
                   decode={Notation.Advanced},
                   encode={Notation.Standard, Notation.MIREX, Notation.Advanced}),
            Symbol("11", {7: Interval.m7, 9: Interval.M9, 11: Interval.P11},
                   decode={Notation.Advanced},
                   encode={Notation.Standard, Notation.MIREX, Notation.Advanced}),
            Symbol("13", {7: Interval.m7, 9: Interval.M9, 11: Interval.P11, 13: Interval.M13},
                   decode={Notation.Advanced},
                   encode={Notation.Standard, Notation.MIREX, Notation.Advanced}),
            Symbol("M9", {7: Interval.M7, 9: Interval.M9},
                   {"maj9", "Maj9", "△9", },
                   decode={Notation.Advanced},
                   encode={Notation.Standard, Notation.MIREX, Notation.Advanced}),
            Symbol("M11", {7: Interval.M7, 9: Interval.M9, 11: Interval.P11},
                   {"maj11", "Maj11", "△11", },
                   decode={Notation.Advanced},
                   encode={Notation.Standard, Notation.MIREX, Notation.Advanced}),
            Symbol("M13", {7: Interval.M7, 9: Interval.M9, 11: Interval.P11, 13: Interval.M13},
                   {"maj13", "Maj13", "△13", },
                   decode={Notation.Advanced},
                   encode={Notation.Standard, Notation.MIREX, Notation.Advanced}),
            Symbol("69", {7: Interval.M6, 9: Interval.M9},
                   decode={Notation.Advanced},
                   encode={Notation.Standard, Notation.Advanced}),

            Symbol("7", {7: Interval.m7},
                   decode={Notation.Standard, Notation.MIREX, Notation.Advanced},
                   encode={Notation.Standard, Notation.MIREX, Notation.Advanced}),
            Symbol("M7", {7: Interval.M7}, {"maj7", "Maj7", "△7", },
                   decode={Notation.Standard, Notation.MIREX, Notation.Advanced},
                   encode={Notation.Standard, Notation.MIREX, Notation.Advanced}),

            Symbol("M", {3: Interval.M3}, {"maj", "Maj", "△", },
                   decode=None, encode={Notation.Standard, Notation.MIREX, Notation.Advanced}),

            Symbol("6", {7: Interval.M6},
                   decode={Notation.Standard, Notation.MIREX, Notation.Advanced},
                   encode={Notation.Standard, Notation.MIREX, Notation.Advanced}),
            # Symbol("6", {7: Interval.M6}, {"M6", "maj6", "Maj6", "△6", "add6", },
            #        decode={Notation.Advanced},
            #        encode={Notation.Advanced}),

            Symbol("sus4", {3: Interval.P4},
                   decode={Notation.Standard, Notation.MIREX, Notation.Advanced},
                   encode={Notation.Standard, Notation.MIREX, Notation.Advanced}),
            Symbol("sus2", {3: Interval.M2},
                   decode={Notation.Standard, Notation.Advanced},
                   encode={Notation.Standard, Notation.Advanced}),

            Symbol("add9", {5: Interval.P5, 7: None, 9: Interval.M9},
                   decode={Notation.Standard, Notation.MIREX, Notation.Advanced},
                   encode={Notation.Standard, Notation.MIREX}),
            Symbol("add9", {9: Interval.M9}, {"add2"},
                   decode=None, encode={Notation.Advanced}),
            Symbol("add11", {11: Interval.P11}, {"add4"},
                   decode=None, encode={Notation.Advanced}),
            Symbol("add13", {13: Interval.M13}, {"add6"},
                   decode=None, encode={Notation.Advanced}),

            Symbol("-5", {5: Interval.d5},
                   decode={Notation.Standard, Notation.MIREX},
                   encode={Notation.Standard, Notation.MIREX}),
            Symbol("-5", {5: Interval.d5}, {"b5", "♭5", },
                   decode={Notation.Advanced},
                   encode={Notation.Advanced}),
            Symbol("+5", {5: Interval.A5},
                   decode={Notation.Standard, Notation.MIREX},
                   encode={Notation.Standard, Notation.MIREX}),
            Symbol("+5", {5: Interval.A5}, {"#5", "♯5", },
                   decode={Notation.Advanced},
                   encode={Notation.Advanced}),

            Symbol("(root)", {3: None, 5: None}, {"omit3,5", "no3,5", "root", },
                   decode={Notation.Standard, Notation.Advanced},
                   encode={Notation.Standard, Notation.Advanced}),
            Symbol("omit5", {5: None}, {"no5", },
                   decode={Notation.Standard, Notation.Advanced},
                   encode={Notation.Standard, Notation.Advanced}),
            Symbol("omit3", {3: None}, {"no3", },
                   decode={Notation.Standard, Notation.Advanced},
                   encode={Notation.Standard, Notation.Advanced}),
            Symbol("omit1", {1: None}, {"omit root", "omitroot", "no1", "no root", "noroot", }, enforced=True,
                   decode={Notation.Standard, Notation.Advanced},
                   encode={Notation.Standard, Notation.Advanced}),
        )

    TensionSuffixDict: Tuple[Symbol, ...] = \
        (
            Symbol("4", {3: Interval.P4}, enforced=True,
                   decode={Notation.MIREX},
                   encode={Notation.MIREX}),
            Symbol("-5", {5: Interval.d5}, {"b5", "♭5", },
                   decode={Notation.Standard, Notation.MIREX, Notation.Advanced},
                   encode={Notation.Standard, Notation.MIREX, Notation.Advanced}),
            Symbol("+5", {5: Interval.A5}, {"#5", "♯5", },
                   decode={Notation.Standard, Notation.MIREX, Notation.Advanced},
                   encode={Notation.Standard, Notation.MIREX, Notation.Advanced}),
            Symbol("b7", {7: Interval.m7}, {"♭7", },
                   decode={Notation.MIREX},
                   encode={Notation.MIREX}),
            Symbol("7", {7: Interval.M7},
                   decode={Notation.MIREX},
                   encode={Notation.MIREX}),
            Symbol("b9", {9: Interval.m9}, {"♭9", },
                   decode={Notation.Standard},
                   encode={Notation.Standard}),
            Symbol("b9", {9: Interval.m9}, {"♭9", "b2", "♭2", },
                   decode={Notation.MIREX, Notation.Advanced},
                   encode={Notation.MIREX, Notation.Advanced}),
            Symbol("9", {9: Interval.M9}, {"♮9", },
                   decode={Notation.Standard},
                   encode={Notation.Standard}),
            Symbol("9", {9: Interval.M9}, {"♮9", "2", "♮2", },
                   decode={Notation.MIREX, Notation.Advanced},
                   encode={Notation.MIREX, Notation.Advanced}),
            Symbol("#9", {9: Interval.m10}, {"♯9", "＃9", },
                   decode={Notation.Standard},
                   encode={Notation.Standard}),
            Symbol("#9", {9: Interval.m10}, {"♯9", "＃9", "#2", "♯2", "＃2", },
                   decode={Notation.MIREX, Notation.Advanced},
                   encode={Notation.MIREX, Notation.Advanced}),
            Symbol("11", {11: Interval.P11}, {"♮11", },
                   decode={Notation.Standard},
                   encode={Notation.Standard}),
            Symbol("11", {11: Interval.P11}, {"♮11", "4", "♮4", },
                   decode={Notation.MIREX, Notation.Advanced},
                   encode={Notation.MIREX, Notation.Advanced}),
            Symbol("#11", {11: Interval.A11}, {"♯11", "＃11"},
                   decode={Notation.Standard},
                   encode={Notation.Standard}),
            Symbol("#11", {11: Interval.A11}, {"♯11", "＃11", "#4", "♯4", "＃4", },
                   decode={Notation.MIREX, Notation.Advanced},
                   encode={Notation.MIREX, Notation.Advanced}),
            Symbol("b13", {13: Interval.m13}, {"♭13", },
                   decode={Notation.Standard},
                   encode={Notation.Standard}),
            Symbol("b13", {13: Interval.m13}, {"♭13", "b6", "♭6", },
                   decode={Notation.MIREX, Notation.Advanced},
                   encode={Notation.MIREX, Notation.Advanced}),
            Symbol("13", {13: Interval.M13}, {"♮13", },
                   decode={Notation.Standard},
                   encode={Notation.Standard}),
            Symbol("13", {13: Interval.M13}, {"♮13", "6", "♮6", },
                   decode={Notation.MIREX, Notation.Advanced},
                   encode={Notation.MIREX, Notation.Advanced}),

            Symbol("*3", {3: None}, enforced=True,
                   decode={Notation.MIREX},
                   encode={Notation.MIREX}),
            Symbol("*5", {5: None}, enforced=True,
                   decode={Notation.MIREX},
                   encode={Notation.MIREX}),
            Symbol("*7", {7: None}, enforced=True,
                   decode={Notation.MIREX},
                   encode={Notation.MIREX}),
            Symbol("1", {3: None, 5: None, 7: None, 9: None, 11: None, 13: None}, enforced=True,
                   decode={Notation.MIREX},
                   encode={Notation.MIREX}),
            Symbol("5", {5: Interval.P5}, enforced=True,
                   decode={Notation.MIREX},
                   encode={Notation.MIREX}),
        )

    TensionParenthesesDict: Dict[Notation, Tuple[Tuple[str, str]]] = \
        {
            Notation.MIREX: (("(", ")"), ),
            Notation.Standard: (("(", ")"), ),
            Notation.Advanced: (("(", ")"), ),
        }

    TensionSeparatorDict: Dict[Notation, Tuple[str, ...]] = \
        {
            Notation.MIREX: (",", ),
            Notation.Standard: (",", ),
            Notation.Advanced: (",", ),
        }

    RootSeparatorDict: Dict[Notation, Tuple[str, ...]] = \
        {
            Notation.MIREX: (":", " ", "", ),
            Notation.Standard: ("", ":", " ", ),
            Notation.Advanced: ("", ":", " ", ),
        }

    BassSeparatorDict: Dict[Notation, Tuple[str, ...]] = \
        {
            Notation.MIREX: ("/", ),
            Notation.Standard: ("/", ),
            Notation.Advanced: ("/", ),
        }

    NoneChordDict: Dict[Notation, Tuple[str, ...]] = \
        {
            Notation.MIREX: ("N", ),
            Notation.Standard: ("N.C.", "N.C", ),
            Notation.Advanced: ("N.C.", "N.C", "NC", "N", ),
        }


DATA: _Data = _Data.get_instance()


class Interpreter:

    @classmethod
    def decode_from_str(cls, value: str, notation: Notation, language: NoteLanguage,
                        *, requires_bass: bool = True) -> Union[Chord, ChordWithBass]:

        for none_chord_str in DATA.NoneChordDict[notation]:
            if none_chord_str.lower() == value.lower():
                if requires_bass:
                    return ChordWithBass(NoneChord)
                else:
                    return NoneChord

        bass: Optional[str] = None
        root: Optional[Note] = None
        chord: str = ""
        tension: str = ""

        # region Divide bass string

        for sep in DATA.BassSeparatorDict[notation]:
            index: int = value.find(sep)
            if index >= 0:
                bass = value[index + len(sep):].lstrip()
                value = value[:index].rstrip()
                break

        # endregion

        # region Divide root string

        tries_auto: bool = False

        for sep in DATA.RootSeparatorDict[notation]:
            if sep:
                index: int = value.find(sep)
                if index >= 0:
                    root = Note.from_str(value[:index], language)
                    chord = value[index + len(sep):].lstrip()
                    break
            else:
                tries_auto = True

        if tries_auto and root is None:
            for index in reversed(range(len(value))):
                try:
                    root = Note.from_str(value[:index + 1], language)
                    chord = value[index + 1:].lstrip()
                    break
                except NotFoundNoteError:
                    pass

        # endregion

        if root is None:
            raise CannotInterpretRootError

        # region Divide chord and tension string

        for l, r in DATA.TensionParenthesesDict[notation]:
            l_index: int = chord.find(l)
            r_index: int = chord.find(r)
            if 0 <= l_index < r_index:
                tension = chord[l_index + len(l): r_index].lstrip()
                chord = chord[:l_index].lstrip()
                break

        # endregion

        chord_dict: Dict[int, Optional[Interval]] = {3: Interval.M3, 5: Interval.P5}

        # region Interpret chord

        altered_3rd: bool = False
        altered_5th: bool = False

        for symbol in DATA.BaseSuffixDict:

            if notation not in symbol.encode_notations:
                continue

            excepted: bool = False
            for s in symbol.excepted:
                if chord.startswith(s):
                    excepted = True
                    break
            if excepted:
                continue

            target_length: int = 0
            for s in symbol.alternatives:
                if chord.startswith(s):
                    target_length = len(s)
                    break
            if target_length == 0:
                if chord.startswith(symbol.symbol):
                    target_length = len(symbol.symbol)
                else:
                    continue

            chord_dict = Interpreter._compose_notes_dict(chord_dict, symbol.intervals,
                                                         altered_3rd, altered_5th, symbol.enforced)
            chord = chord[target_length:].lstrip()
            altered_3rd = altered_3rd or 3 in symbol.intervals
            altered_5th = altered_5th or 5 in symbol.intervals

            if not chord:
                break

        # endregion

        if chord:
            raise CannotEncodeWholeChordStringError(chord)

        # region Interpret tension

        for symbol in DATA.TensionSuffixDict:

            if notation not in symbol.encode_notations:
                continue

            excepted: bool = False
            for s in symbol.excepted:
                if chord.startswith(s):
                    excepted = True
                    break
            if excepted:
                continue

            target_length: int = 0
            for s in symbol.alternatives:
                if tension.startswith(s):
                    target_length = len(s)
                    break
            if target_length == 0:
                if tension.startswith(symbol.symbol):
                    target_length = len(symbol.symbol)
                else:
                    continue

            chord_dict = Interpreter._compose_notes_dict(chord_dict, symbol.intervals,
                                                         altered_3rd, altered_5th, symbol.enforced)
            tension = tension[target_length:].lstrip()
            while True:
                sep_length: int = 0
                for sep in DATA.TensionSeparatorDict[notation]:
                    if tension.startswith(sep):
                        sep_length = max(sep_length, len(sep))
                tension = tension[sep_length:].lstrip()
                if sep_length == 0:
                    break

            if not tension:
                break

        # endregion

        if tension:
            raise CannotEncodeWholeTensionStringError(tension)

        # if chord_dict.get(7, None) == Interval.A7:
        #     chord_dict[7] = None
        # if chord_dict.get(9, None) == Interval.d9:
        #     chord_dict[9] = None

        created_chord: Chord = Chord.from_dict(chord_dict, root_note=root,
                                               omits_root=chord_dict.get(1, Interval.P1) is None, language=language)

        # created_chord: Chord = Chord(root, chord_dict.get(3, None), chord_dict.get(5, None),
        #                              chord_dict.get(7, None), chord_dict.get(9, None),
        #                              chord_dict.get(11, None), chord_dict.get(13, None),
        #                              omits_root=chord_dict.get(1, Interval.P1) is None, language=language)

        if requires_bass:
            return ChordWithBass(created_chord, bass, notation=notation, language=language)
        else:
            return created_chord

    @classmethod
    def encode_to_str(cls, value: Union[Chord, ChordWithBass], notation: Notation) -> str:

        # noinspection PyUnusedLocal
        chord: Chord

        if type(value) == Chord:
            chord = value
        elif type(value) == ChordWithBass:
            chord = value.chord
        else:
            raise CannotDecodeChordError

        if chord == NoneChord:
            return DATA.NoneChordDict[notation][0]

        chord_str: str = ""

        chord_dict: Dict[int, Optional[Interval]] = {1: Interval.P1, 3: Interval.M3, 5: Interval.P5}
        # chord_dict: Dict[int, Optional[Note]] = {
        #     1: value.root,
        #     3: Note.from_int(value.root + Interval.M3),
        #     5: Note.from_int(value.root + Interval.P5)}
        altered_3rd: bool = False
        altered_5th: bool = False

        for symbol in DATA.BaseSuffixDict:

            if chord_dict == chord:
                break
            if notation not in symbol.decode_notations:
                continue
            if any(chord.intervals[k] != v for k, v in symbol.intervals.items()):
                continue

            chord_dict = Interpreter._compose_notes_dict(
                chord_dict, symbol.intervals, altered_3rd, altered_5th, symbol.enforced)

            altered_3rd = altered_3rd or 3 in symbol.intervals
            altered_5th = altered_5th or 5 in symbol.intervals
            chord_str += symbol.symbol

        tension_str: str = ""

        for symbol in DATA.TensionSuffixDict:

            if chord_dict == chord:
                break
            if notation not in symbol.decode_notations:
                continue
            if any(chord.intervals[k] != v for k, v in symbol.intervals.items()):
                continue

            chord_dict = Interpreter._compose_notes_dict(
                chord_dict, symbol.intervals, altered_3rd, altered_5th, symbol.enforced)

            altered_3rd = altered_3rd or 3 in symbol.intervals
            altered_5th = altered_5th or 5 in symbol.intervals
            if tension_str:
                tension_str += DATA.TensionSeparatorDict[notation][0]
            tension_str += symbol.symbol

        if chord_dict != chord:
            raise CannotDecodeChordError

        full_str: str = str(chord.root)
        full_str += DATA.RootSeparatorDict[notation][0]
        full_str += chord_str
        if tension_str:
            full_str += DATA.TensionParenthesesDict[notation][0][0]
            full_str += tension_str
            full_str += DATA.TensionParenthesesDict[notation][0][1]
        if type(value) == ChordWithBass and value.bass != chord.root:
            full_str += DATA.BassSeparatorDict[notation][0]
            full_str += str(value.bass)
        return full_str

    @classmethod
    def _compose_notes_dict(cls, base: Dict[int, T], composed: Dict[int, T],
                            altered_3rd: bool, altered_5th: bool, enforced: bool) -> Dict[int, T]:

        # noinspection PyUnusedLocal
        k: int
        # noinspection PyUnusedLocal
        v: T

        for k, v in composed.items():
            if k in base and not (enforced or (k == 3 and not altered_3rd) or (k == 5 and not altered_5th)):
                raise CollideNoteError
            base[k] = v

        return base
