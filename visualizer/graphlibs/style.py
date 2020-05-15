from enum import auto, Enum, unique


@unique
class LineStyle(Enum):

    Unknown = 0

    Nothing = auto()

    Solid = auto()
    Dashed = auto()
    Dotted = auto()
    DashDot = auto()


class MarkerStyle(Enum):

    Unknown = 0

    Nothing = auto()

    Dot = auto()
    Pixel = auto()
    Circle = auto()

    Square = auto()
    Triangle = auto()
    TriangleUp = Triangle
    Diamond = auto()

    Pentagon = auto()
    Hexagon1 = auto()
    Hexagon2 = auto()
    Octagon = auto()

    FilledPlus = auto()
    FilledCross = auto()
    Star = auto()
    DiamondThin = auto()

    TriangleDown = auto()
    TriangleLeft = auto()
    TriangleRight = auto()

    Plus = auto()
    Cross = auto()

    TriUp = auto()
    TriDown = auto()
    TriLeft = auto()
    TriRight = auto()

    VLine = auto()
    HLine = auto()

    TickUp = auto()
    TickDown = auto()
    TickLeft = auto()
    TickRight = auto()

    CaretUp = auto()
    CaretDown = auto()
    CaretLeft = auto()
    CaretRight = auto()

    CaretBasedUp = auto()
    CaretBasedDown = auto()
    CaretBasedLeft = auto()
    CaretBasedRight = auto()

    Function = auto()
