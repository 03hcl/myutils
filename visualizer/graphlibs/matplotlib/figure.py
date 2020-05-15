from types import TracebackType
from typing import Optional, Tuple, Type, TypeVar

from matplotlib.axes import Axes as AxesRaw
from matplotlib.figure import Figure as FigureRaw

from .. import CannotShowFigureError, FigureBase, PropertyBase
from ..io import create_full_path

from . import Axes

_figure = TypeVar("_figure", bound="Figure")


class Figure(FigureBase):

    class WindowTitle(PropertyBase[str]):

        def __init__(self, figure: FigureRaw, *args, **kwargs):
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            self._canvas: FigureCanvas = figure.canvas
            super(Figure.WindowTitle, self).__init__(*args, **kwargs)

        def _get(self, **kwargs) -> str:
            return self._canvas.get_window_title()

        def _set(self, value: str, **kwargs) -> None:
            self._canvas.set_window_title(title=value)

    def __init__(self, grid: Optional[Tuple[int, int]] = None, *args, **kwargs):

        self._raw: Optional[FigureRaw] = None
        self._figure_number: int = 0
        self._grid: Optional[Tuple[int, int]] = grid

        self._window_title: Optional[Figure.WindowTitle] = None

        super(Figure, self).__init__(*args, **kwargs)

    # region Special Method

    def __enter__(self) -> FigureBase:
        if self._grid is None:
            self._grid = (1, 1)
        # if self._grid[0] <= 0 or self._grid[1] <= 0:
        #     raise InvalidGridError
        self.open(self._grid, *self._init_args, **self._init_kwargs)
        return self

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_val: Optional[BaseException],
                 exc_tb: Optional[TracebackType]) -> None:
        self.close()

    # endregion

    # region FigureBase Method

    def open(self, grid: Tuple[int, int] = (1, 1), *args, **kwargs) -> FigureRaw:
        import matplotlib.pyplot as plt
        if self._figure_number > 0:
            self.close()
        self._raw: FigureRaw = plt.figure(*args, **kwargs)
        self._figure_number = self._raw.number
        self._grid = grid
        self._window_title = Figure.WindowTitle(self._raw)
        return self._raw

    def close(self) -> None:
        import matplotlib.pyplot as plt
        plt.close(self._figure_number)
        self._figure_number: int = 0
        self._grid = (0, 0)
        self._window_title = None

    def window_title(self, value: Optional[str]) -> Optional[str]:
        return None if self._window_title is None else self._window_title(value)

    def axes(self, location: Tuple[int, int] = (0, 0), span: Tuple[int, int] = (1, 1)) -> Axes:
        import matplotlib.pyplot as plt
        raw: AxesRaw = plt.subplot2grid(shape=self._grid, loc=location, rowspan=span[0], colspan=span[1])
        return Axes(raw, self)

    def save_as_png(self, directory_path: str, file_name: str) -> None:
        self._raw.savefig(create_full_path(directory_path, file_name, "png"))

    def show(self) -> None:
        import matplotlib.pyplot as plt
        if not plt.fignum_exists(self._figure_number):
            raise CannotShowFigureError
        # plt.draw()
        plt.show()

        # plt.ion()
        # self._raw.show()
        # self._raw.draw()
        # plt.pause(0.1)
        # plt.draw()
        # plt.ioff()

    # endregion

    # region GraphBase Method

    def clear(self) -> None:
        self._raw.clf()

    # endregion

    # region Private Method

    def _select(self) -> FigureRaw:
        import matplotlib.pyplot as plt
        return plt.figure(self._figure_number)

    # endregion
