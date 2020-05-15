from abc import abstractmethod
from typing import Optional, Tuple

from .graph_base import GraphBase


class ColorBar(GraphBase):

    def __init__(self, *args, **kwargs):
        super(ColorBar, self).__init__(*args, **kwargs)

    @abstractmethod
    def range(self, value: Optional[Tuple[Optional[float], Optional[float]]] = None, **kwargs) -> Tuple[float, float]:
        pass


ColorBarBase = ColorBar
