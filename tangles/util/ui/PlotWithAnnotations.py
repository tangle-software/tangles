import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from typing import Callable, Optional, Union


AnnotationFuncImg = Callable[[float],np.ndarray]
"""A function providing content for an annotation of type image (2d-np.ndarray)."""

AnnotationFuncTxt = Callable[[float],str]
"""A function providing content for an annotation of type text."""

AnnotationFuncAx = Callable[[mpl.axes.Axes, float], np.ndarray]
"""A function providing content for an annotation by drawing inside an matplotlib.Axes object."""

def wrap_annotation_func_ax(annotation_func_ax: AnnotationFuncAx) -> AnnotationFuncImg:
    """Create an AnnotationFuncImg that calls the given AnnotationFuncAx.

    Parameters
    ----------
    annotation_func_ax: AnnotationFuncAx
        Your annotation function taking an ax object.

    Returns
    -------
    AnnotationFuncImg
        An annotation function that can be used in :func:`plot_annotated`.
    """
    
    canvas = FigureCanvasAgg(mpl.figure.Figure(figsize=(1, 1)))
    ax = canvas.figure.add_axes([0,0,1,1])
    ax.axis('off')
    artists = []    # for some reason this version is significantly faster than any other way to clear the axes in between (that I tried ...) -> annotation_func has to return a list of the created artists
    def func(x):
        for a in artists: a.remove()
        artists.clear()
        artists.extend(annotation_func_ax(ax,x))
        canvas.draw()
        return np.array(canvas.renderer.buffer_rgba())
    return func


def plot_annotated(x, y, annotation_x_positions, annotation_offsets, annotation_func:AnnotationFuncImg|AnnotationFuncTxt, annotation_is_image:bool=False, annotation_zoom:float=1,
                   label:str=None, title:str=None, xlabel:str=None, ylabel:str=None, figsize:tuple[float,float]=(10,5), ax:Optional[mpl.axes.Axes]=None, interactive:bool=False):
    """A plot with annotations that can either be images or texts. Annotations can be provided by the caller through a callback function.

    Parameters
    ----------
    x : arraylike
        x values to plot.
    y : arraylike
        y values to plot.
    annotation_x_positions : arraylike
        x-values to be annotations (in units of the x-axis).
    annotation_offsets : arraylike
        Array of 2-tuples. Each entry specifies the offset ``(dx, dy)`` of the annotation box from the
        `annotation_x_positions` (in pixels).
    annotation_func : AnnotationFuncImg or AnnotationFuncTxt
        Function that provides the content of the annotation box. The content can either be an image or a string. 
        If you want to use a matplotlib.axes.Axes object see :meth:`wrap_annotation_func_ax`.
    annotation_is_image : bool
        Pass True if `annotation_func` is of type AnnotationFuncImg.
    annotation_zoom : float
        Scaling applied to the annotations.
    label : str
        Label of the curve.
    title : str
        Title of the plot.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    figsize : 2-tuple of float
        Figure size (if parameter `ax` is not present).
    ax : matplotlib.Axes
        The ax to plot into.
    interactive : bool
        If True, the plot is interactive and shows an annotation at the mouse position 
        whenever the mouse is clicked or dragged.
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    if not isinstance(annotation_x_positions, np.ndarray):
        annotation_x_positions = np.asarray(annotation_x_positions)

    ax.plot(x, y, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(annotation_x_positions)

    for anno_x, (box_x, box_y) in zip(annotation_x_positions, annotation_offsets):
        data_idx = (np.abs(x - anno_x)).argmin()
        _create_annotation(ax, anno_x, y[data_idx], annotation_is_image, annotation_zoom, annotation_func, box_x, box_y)

    if interactive:
        dyn_objects = []
        def update_annotations(dyn_annotation_x=None):
            for do in dyn_objects: do.remove()
            dyn_objects.clear()

            if dyn_annotation_x:
                data_y = y[(np.abs(x - dyn_annotation_x)).argmin()]
                box_x, box_y = _interpolate_box_pos(dyn_annotation_x, annotation_x_positions, annotation_offsets)

                dyn_objects.append(_create_annotation(ax, dyn_annotation_x, data_y, annotation_is_image, annotation_zoom, annotation_func, box_x, box_y))
        mouse_down_ptr = [False]
        def mouse(event):
            if event.inaxes == ax:
                mouse_down_ptr[0] = event.name == 'button_press_event' or (mouse_down_ptr[0] and event.name != 'button_release_event')
                if mouse_down_ptr[0]:
                    update_annotations(event.xdata)
                    fig.canvas.draw_idle()
        fig.canvas.mpl_connect('button_press_event', mouse)
        fig.canvas.mpl_connect('button_release_event', mouse)
        fig.canvas.mpl_connect('motion_notify_event', mouse)


def _interpolate_box_pos(dyn_annotation_x, annotation_x_positions, annotation_offsets):
    if annotation_offsets is None or len(annotation_offsets)==0:
        return None, None
    if len(annotation_offsets)==1:
        return annotation_offsets[0]

    right_x_pos = annotation_x_positions >= dyn_annotation_x
    if not right_x_pos.any():
        return annotation_offsets[-1]
    right_idx = right_x_pos.argmax()
    if right_idx == 0:
        return annotation_offsets[0]
    left_idx = right_idx - 1

    d_total = annotation_x_positions[right_idx] - annotation_x_positions[left_idx]
    w_left, w_right = 1 - (dyn_annotation_x - annotation_x_positions[left_idx]) / d_total, 1 - (annotation_x_positions[right_idx] - dyn_annotation_x) / d_total
    left_x_offset, left_y_offset = annotation_offsets[left_idx]
    right_x_offset, right_y_offset = annotation_offsets[right_idx]
    return (w_left * left_x_offset + w_right * right_x_offset), (w_left * left_y_offset + w_right * right_y_offset)


def _create_annotation(ax, data_x, data_y, annotation_is_image, annotation_zoom, annotation_func,
                       box_x=None, box_y=None):
    if annotation_is_image:
        off = OffsetImage(annotation_func(data_x), zoom=annotation_zoom)
    else:
        off = TextArea(annotation_func(data_x))

    if box_x is None: box_x = 0
    if box_y is None: box_y = 100

    ab = AnnotationBbox(off, (data_x, data_y), xybox=(box_x, box_y), xycoords='data', boxcoords="offset pixels",
                        arrowprops=dict(arrowstyle="->"), bboxprops=dict(boxstyle="round"), frameon=True)
    off.axes = ax
    ax.add_artist(ab)
    return ab


