from typing import Optional, Tuple, Type

import numpy as np

from .graphlibs import Axes, Axis, Figure
from .graphlibs.matplotlib import Figure as MPLFigure


def visualize_values_for_each_epoch(
        loss_array: np.ndarray, keys: Tuple[str, ...],
        *, graph_library: Type[Figure] = MPLFigure, directory: str = "", file_name: str,
        pre_epoch: int, y_axis_name: Optional[str] = None,
        is_logscale: bool = False, data_range: Optional[Tuple[Optional[float], Optional[float]]] = None) -> None:

    with graph_library(figsize=(20 / 3, 15 / 4), dpi=96) as graph:
        ax: Axes = graph.axes()
        axis_x: Axis = ax.axis_x()
        axis_x.label("epoch")
        axis_x.range((pre_epoch, max(loss_array[:, 0])))
        axis_y: Axis = ax.axis_y()
        axis_y.label(file_name if y_axis_name is None else y_axis_name)
        if is_logscale:
            axis_y.scale(Axis.Scale.Log)
        else:
            axis_y.range(data_range or (0, max(loss_array[:, 1:].flatten())))
        for i in range(1, loss_array.shape[1]):
            ax.line(loss_array[:, 0], loss_array[:, i], label=keys[i - 1])
        ax.legend(len(keys) > 1)
        graph.save_as_png(directory, file_name)


def visualize_loss(loss_array: np.ndarray, keys: Tuple[str, ...],
                   *, graph_library: Type[Figure] = MPLFigure, directory: str = "", file_name: str = "loss",
                   pre_epoch: int = 0,
                   is_logscale: bool = False, data_range: Optional[Tuple[Optional[float], Optional[float]]] = None) \
        -> None:
    visualize_values_for_each_epoch(loss_array, keys,
                                    graph_library=graph_library, directory=directory, file_name=file_name,
                                    pre_epoch=pre_epoch, y_axis_name="loss",
                                    is_logscale=is_logscale, data_range=data_range)
