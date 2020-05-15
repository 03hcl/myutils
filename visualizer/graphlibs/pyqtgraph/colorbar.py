from abc import ABC
from typing import Optional, Tuple

import numpy as np

from pyqtgraph import GraphicsLayout, HistogramLUTItem, ImageItem

from .. import ColorBarBase, PropertyBase


class ColorBar(ColorBarBase):

    class Property(PropertyBase, ABC):

        def __init__(self, color_bar: HistogramLUTItem, image: ImageItem, *args, **kwargs):
            self._color_bar: HistogramLUTItem = color_bar
            self._image: ImageItem = image
            super(ColorBar.Property, self).__init__(*args, **kwargs)

    class _Range(Property, PropertyBase[Tuple[Optional[float], Optional[float]]]):

        def _get(self, **kwargs) -> Tuple[Optional[float], Optional[float]]:
            return self._color_bar.axis.range

        def _set(self, value: Tuple[Optional[float], Optional[float]], **kwargs) -> None:
            value = (self._color_bar.axis.range[0] if value[0] is None else value[0],
                     self._color_bar.axis.range[1] if value[1] is None else value[1])
            value = (np.min(self._image.image) if value[0] == -np.inf else value[0],
                     np.max(self._image.image) if value[1] == np.inf else value[1])
            if "padding" not in kwargs:
                kwargs["padding"] = 0
            self._color_bar.setHistogramRange(mn=value[0], mx=value[1], **kwargs)

    def __init__(self, color_bar: HistogramLUTItem, figure: GraphicsLayout, image: ImageItem, *args, **kwargs):

        self._raw: HistogramLUTItem = color_bar
        self._figure: GraphicsLayout = figure

        self._range: ColorBar._Range = ColorBar._Range(self._raw, image)
        # self._range: ColorBar._Range = ColorBar._Range(self._raw, alt=(np.nan, np.nan))

        self._raw.setImageItem(image)
        super(ColorBar, self).__init__(*args, **kwargs)

    def range(self, value: Optional[Tuple[Optional[float], Optional[float]]] = None, **kwargs) -> Tuple[float, float]:
        return self._range(value, **kwargs)

    def clear(self) -> None:
        self._figure.removeItem(self._raw)
