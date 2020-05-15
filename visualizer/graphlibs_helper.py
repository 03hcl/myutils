from enum import auto, Enum, unique

# from .graphlibs import Figure
# from .graphlibs import matplotlib


@unique
class MyStyle(Enum):

    Unknown = 0

    SansSerif_Helvetica = auto()
    SansSerif_Arial = auto()
    SansSerif_NotoSans = auto()
    SansSerif_Roboto = auto()
    SansSerif_Cambria = auto()
    SansSerif_Segoe = auto()
    SansSerif_Verdana = auto()
    SansSerif_HiraginoKakuGothic = auto()
    SansSerif_HiraginoMaruGothic = auto()
    SansSerif_YuGothic = auto()

    Serif_TimesNewRoman = auto()
    Serif_NotoSerif = auto()
    Serif_PTSerif = auto()
    Serif_PalatinoLinotype = auto()
    Serif_Century = auto()
    Serif_Georgia = auto()
    Serif_HiraginoMincho = auto()
    Serif_YuMincho = auto()
    Serif_Meiryo = auto()

    MonoSpace_CourierNew = auto()
    MonoSpace_Consolas = auto()
    MonoSpace_SourceHanCode = auto()
    MonoSpace_RictyDiscord = auto()
    MonoSpace_Inconsolata = auto()
    MonoSpace_Monaco = auto()
    MonoSpace_RobotoMono = auto()

    def __str__(self):
        if self == MyStyle.Unknown:
            return ""
        if self == MyStyle.SansSerif_Helvetica:
            return ""
        if self == MyStyle.SansSerif_Arial:
            return ""
        if self == MyStyle.SansSerif_NotoSans:
            return ""
        if self == MyStyle.SansSerif_Roboto:
            return ""
        if self == MyStyle.SansSerif_Cambria:
            return ""
        if self == MyStyle.SansSerif_Segoe:
            return ""
        if self == MyStyle.SansSerif_Verdana:
            return ""
        if self == MyStyle.SansSerif_HiraginoKakuGothic:
            return ""
        if self == MyStyle.SansSerif_HiraginoMaruGothic:
            return ""
        if self == MyStyle.SansSerif_YuGothic:
            return ""
        if self == MyStyle.Serif_TimesNewRoman:
            return ""
        if self == MyStyle.Serif_NotoSerif:
            return ""
        if self == MyStyle.Serif_PTSerif:
            return ""
        if self == MyStyle.Serif_PalatinoLinotype:
            return ""
        if self == MyStyle.Serif_Century:
            return ""
        if self == MyStyle.Serif_Georgia:
            return ""
        if self == MyStyle.Serif_HiraginoMincho:
            return ""
        if self == MyStyle.Serif_YuMincho:
            return ""
        if self == MyStyle.Serif_Meiryo:
            return ""
        if self == MyStyle.MonoSpace_CourierNew:
            return ""
        if self == MyStyle.MonoSpace_Consolas:
            return ""
        if self == MyStyle.MonoSpace_SourceHanCode:
            return ""
        if self == MyStyle.MonoSpace_RictyDiscord:
            return ""
        if self == MyStyle.MonoSpace_Inconsolata:
            return ""
        if self == MyStyle.MonoSpace_Monaco:
            return ""
        if self == MyStyle.MonoSpace_RobotoMono:
            return ""
        raise NotImplementedError


def create_styled_figure():
    pass
