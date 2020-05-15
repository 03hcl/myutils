from typing import Optional, Union

from ..interval import Interval, IntervalInterpreter
from ..interval.exceptions import NotAnIntervalError
from ..note import Note, NoteLanguage
from ..note.exceptions import NotFoundNoteError

from .chord import Chord, NoneChord, NoteLike
from .notation import Notation


class ChordWithBass:

    # region Property

    @property
    def chord(self) -> Chord:
        return self._chord

    @property
    def bass(self) -> Optional[Note]:
        return self._bass

    @property
    def contains_bass_in_chord(self) -> bool:
        return self._contains_bass_in_chord

    @property
    def inverted(self) -> bool:
        return self._inverted

    # endregion

    def __init__(self, chord: Chord, bass: NoteLike = None,
                 *, notation: Notation = Notation.Standard, language: Union[str, NoteLanguage] = NoteLanguage.MIDI):

        self._chord: Chord = chord
        self._bass: Optional[Note] = None

        if type(bass) == Note:
            self._bass = bass
        elif type(bass) == Interval:
            self._bass = Note.from_int(self.chord.root.value + bass)
        elif type(bass) == str:
            try:
                self._bass = Note.from_str(bass, language)
            except NotFoundNoteError:
                try:
                    interval: Interval = IntervalInterpreter.to_interval_mirex(bass) \
                        if notation == Notation.MIREX else IntervalInterpreter.to_interval(bass)
                    self._bass = Note.from_int(self.chord.root.value + interval)
                except NotAnIntervalError:
                    try:
                        self._bass = self.chord[int(bass)]
                    finally:
                        pass

        self._contains_bass_in_chord: bool = False

        # noinspection PyUnusedLocal
        n: Optional[Note]
        for n in self.chord.values():
            if self._bass is None:
                self._bass = n
            if self._bass is not None and n == self._bass:
                self._contains_bass_in_chord = True
                break

        self._inverted: bool = self._contains_bass_in_chord and self._bass != self.chord.root

    def __str__(self):
        if self._inverted or (not self._contains_bass_in_chord and self.chord != NoneChord):
            return "{} / {}".format(str(self._chord), str(self._bass))
        else:
            return str(self._chord)
