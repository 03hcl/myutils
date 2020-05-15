from types import TracebackType
from typing import Dict, Tuple, Optional, Type, TypeVar

from pyqtgraph import GraphicsLayout, GraphicsView, GraphicsWindow  # , GraphicsLayoutWidget
from pyqtgraph.exporters import ImageExporter
from PyQt5.QtWidgets import QApplication, QMainWindow

from .. import CannotShowFigureError, FigureBase, PropertyBase
from ..io import create_full_path

from . import Axes

_figure = TypeVar("_figure", bound="Figure")


class Figure(FigureBase):

    class WindowTitle(PropertyBase[str]):

        def __init__(self, window: QMainWindow, *args, **kwargs):
            self._window: QMainWindow = window
            super(Figure.WindowTitle, self).__init__(*args, **kwargs)

        def _get(self, **kwargs) -> str:
            return self._window.windowTitle()

        def _set(self, value: str, **kwargs) -> None:
            self._window.setWindowTitle(value)

    def __init__(self, grid: Optional[Tuple[int, int]] = None, *args, **kwargs):
        # self._app: QApplication = QApplication([])
        # app = self._app
        self._view: Optional[GraphicsView] = None
        self._raw: Optional[GraphicsLayout] = None

        self._grid: Optional[Tuple[int, int]] = grid
        self._axes: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Axes] = {}
        self._window_title: Optional[Figure.WindowTitle] = None

        super(Figure, self).__init__(*args, **kwargs)

    # region Special Method

    def __enter__(self) -> _figure:
        if self._grid is None:
            self._grid = (1, 1)
        # if self._grid[0] <= 0 or self._grid[1] <= 0:
        #     raise InvalidGridError
        self.open(self._grid, *self._init_args, **self._init_kwargs)
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException],
                 exc_tb: Optional[TracebackType]) -> None:
        self.close()

    # endregion

    # region FigureBase Method

    def open(self, grid: Tuple[int, int] = (1, 1), *args, **kwargs) -> GraphicsLayout:
        # self._view = GraphicsLayoutWidget()
        self._view = GraphicsWindow()
        self._raw = self._view.centralWidget

        self._grid = grid
        self._init_axis(self._grid)
        self._window_title = Figure.WindowTitle(self._view)
        return self._raw

    def close(self) -> None:
        self._view.close()
        # self._app.closeAllWindows()
        self._view = None
        self._grid = (0, 0)
        self._axes = {}
        self._window_title = None

    def window_title(self, value: Optional[str]) -> Optional[str]:
        return None if self._window_title is None else self._window_title(value)

    def axes(self, location: Tuple[int, int] = (0, 0), span: Tuple[int, int] = (1, 1)) -> Axes:
        if not (location, span) in self._axes:
            self._add_axes(location, span)
        return self._axes[(location, span)]

    def save_as_png(self, directory_path: str, file_name: str) -> None:
        QApplication.processEvents()
        exporter = FixedImageExporter(self._view.sceneObj)
        # exporter.parameters()["width"] = 1280
        # exporter.parameters()["height"] = 960
        exporter.export(create_full_path(directory_path, file_name, "png"))

    def show(self) -> None:
        if self._view is None:
            raise CannotShowFigureError
        # win = QMainWindow()
        # win.setCentralWidget(self._view)
        # win.show()
        # win.setWindowTitle(self._window_title)
        # self._app.exec_()
        QApplication.instance().exec_()

    # endregion

    # region GraphBase Method

    def clear(self) -> None:
        for a in self._axes.values():
            a.remove_from_figure()
        self._init_axis(self._grid)

    # endregion

    # region Private Method

    def _add_axes(self, location: Tuple[int, int], span: Tuple[int, int]) -> None:
        view = self._raw.addLayout(row=location[0], col=location[1], rowspan=span[0], colspan=span[1])
        self._axes[(location, span)] = Axes(view, self._raw)

    def _init_axis(self, grid: Tuple[int, int]) -> None:
        for row in range(grid[0]):
            for column in range(grid[1]):
                self._add_axes(location=(row, column), span=(1, 1))

    # endregion


class FixedImageExporter(ImageExporter):

    def widthChanged(self):
        sr = self.getSourceRect()
        ar = float(sr.height()) / sr.width()
        self.params.param("height").setValue(int(self.params["width"] * ar), blockSignal=self.heightChanged)

    def heightChanged(self):
        sr = self.getSourceRect()
        ar = float(sr.width()) / sr.height()
        self.params.param("width").setValue(int(self.params["height"] * ar), blockSignal=self.widthChanged)
