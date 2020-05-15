from .. import LineStyle, MarkerStyle


def line_style_to_str(self):
    if self == LineStyle.Unknown:
        return ""
    if self == LineStyle.Solid:
        return "-"
    if self == LineStyle.Dashed:
        return "--"
    if self == LineStyle.DashDot:
        return "-."
    if self == LineStyle.Dotted:
        return ":"
    raise NotImplementedError


def marker_style_to_str(self):
    if self == MarkerStyle.Unknown:
        return ""
    if self == MarkerStyle.Nothing:
        return " "
    if self == MarkerStyle.Dot:
        return "."
    if self == MarkerStyle.Pixel:
        return ","
    if self == MarkerStyle.Circle:
        return "o"
    if self == MarkerStyle.Square:
        return "s"
    if self == MarkerStyle.Triangle:
        return "^"
    if self == MarkerStyle.Diamond:
        return "D"
    if self == MarkerStyle.Pentagon:
        return "p"
    if self == MarkerStyle.Hexagon1:
        return "h"
    if self == MarkerStyle.Hexagon2:
        return "H"
    if self == MarkerStyle.Octagon:
        return "8"
    if self == MarkerStyle.FilledPlus:
        return "P"
    if self == MarkerStyle.FilledCross:
        return "X"
    if self == MarkerStyle.Star:
        return "*"
    if self == MarkerStyle.DiamondThin:
        return "d"
    if self == MarkerStyle.TriangleDown:
        return "v"
    if self == MarkerStyle.TriangleLeft:
        return "<"
    if self == MarkerStyle.TriangleRight:
        return ">"
    if self == MarkerStyle.Plus:
        return "+"
    if self == MarkerStyle.Cross:
        return "x"
    if self == MarkerStyle.TriUp:
        return "1"
    if self == MarkerStyle.TriDown:
        return "2"
    if self == MarkerStyle.TriLeft:
        return "3"
    if self == MarkerStyle.TriRight:
        return "4"
    if self == MarkerStyle.VLine:
        return "|"
    if self == MarkerStyle.HLine:
        return "_"
    raise NotImplementedError
