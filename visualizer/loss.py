from typing import Tuple, Type

import numpy as np

from .graphlibs import Axes, Axis, FigureBase
from .graphlibs.matplotlib import Figure


def visualize_loss(loss_array: np.ndarray, keys: Tuple[str, ...],
                   *, graph_library: Type[FigureBase] = Figure,
                   directory: str = "", file_name: str = "loss",
                   pre_epoch: int = 0, is_logscale: bool = False) -> None:

    with graph_library(figsize=(20 / 3, 15 / 4), dpi=96) as graph:
        ax: Axes = graph.axes()
        axis_x: Axis = ax.axis_x()
        axis_x.label("epoch")
        axis_x.range((pre_epoch, max(loss_array[:, 0])))
        axis_y: Axis = ax.axis_y()
        axis_y.label("loss")
        if is_logscale:
            axis_y.scale(Axis.Scale.Log)
        else:
            axis_y.range((0, max(loss_array[:, 1:].flatten())))
        for i in range(1, loss_array.shape[1]):
            ax.line(loss_array[:, 0], loss_array[:, i], label=keys[i - 1])
        ax.legend(len(keys) > 1)
        graph.save_as_png(directory, file_name)
