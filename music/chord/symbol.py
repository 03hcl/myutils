from typing import Dict, Optional, Set

from ..interval import Interval

from .notation import Notation


class Symbol:
    def __init__(self, symbol: str, intervals: Dict[int, Optional[Interval]],
                 alternatives: Optional[Set[str]] = None,
                 *, excepted: Optional[Set[str]] = None, enforced: bool = False,
                 decode: Optional[Set[Notation]], encode: Optional[Set[Notation]]):
        self.symbol: str = symbol
        self.intervals: Dict[int, Optional[Interval]] = intervals
        self.alternatives: Set[str] = alternatives or {}
        self.excepted: Set[str] = excepted or {}
        self.enforced: bool = enforced
        self.decode_notations: Set[Notation] = decode or {}
        self.encode_notations: Set[Notation] = encode or {}
