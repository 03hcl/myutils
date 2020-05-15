from .. import LineStyle, MarkerStyle


def line_style_to_dict(self):
    if self == LineStyle.Unknown:
        return {}
    if self == LineStyle.Solid:
        raise NotImplementedError
    if self == LineStyle.Dashed:
        raise NotImplementedError
    if self == LineStyle.DashDot:
        raise NotImplementedError
    if self == LineStyle.Dotted:
        raise NotImplementedError
    raise NotImplementedError


def marker_style_to_dict(self):
    if self == MarkerStyle.Unknown:
        return {}
    if self == MarkerStyle.Nothing:
        return {"symbol": None}
    if self == MarkerStyle.Dot:
        return {"symbol": "o", "symbolSize": 6}
    if self == MarkerStyle.Pixel:
        return {"symbol": "o", "symbolSize": 2.5}
    if self == MarkerStyle.Circle:
        return {"symbol": "o", "symbolSize": 10}
    if self == MarkerStyle.Square:
        return {"symbol": "s", "symbolSize": 10}
    if self == MarkerStyle.Triangle:
        raise NotImplementedError
    if self == MarkerStyle.Diamond:
        raise NotImplementedError
    if self == MarkerStyle.DiamondThin:
        return {"symbol": "d", "symbolSize": 10}
    if self == MarkerStyle.TriangleDown:
        return {"symbol": "t", "symbolSize": 10}
    if self == MarkerStyle.Plus:
        return {"symbol": "+", "symbolSize": 10}
    raise NotImplementedError
