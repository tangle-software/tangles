import matplotlib.pyplot as plt
import numpy as np


class XYPicker:
    """
    Interactive plot, which allows the user to pick different points from a two-dimensional plot
    and interactively see the results. Based on the matplotlib library.

    Parameters
    ----------
    ax
        A matplotlib axis object.
    sel_callback
        A function which takes a x and a y coordinate and optionally another object.
        The function is called whenever the user makes a selection and is given the
        x- and y-coordinates which the user selected.
        If you want the function to take an object you have to set the `callback_object`
        attribute of the XYPicker instance.
    xrange : list
        Specify an interval of the form ``[min_x_value, max_x_value]``.
    yrange : list
        Specify an interval of the form ``[min_y_value, max_y_value]``.
    xticks : int, list, np.ndarray or range
        Either a list, np.ndarray or range containing the ticks of the x-axis
        or an int determining the number of ticks on the x-axis. Defaults to 10.
    yticks : int, list, np.ndarray or range
        Either a list, np.ndarray or range containing the ticks of the y-axis
        or an int determining the number of ticks on the y-axis. Defaults to 10.
    continuous_update : bool
        Whether to allow drag and drop or only to click on the plot.
    draw_custom_picker
        A function that takes the axis object and a tuple ``(selected_x, selected_y)``
        of the selected point on the plot.
        Allows the user to draw a different picker, instead of the default plot.

    Attributes
    ----------
    xlabel : str
        The label of the x-axis. Defaults to None.
    ylabel : str
        The label of the y-axis. Defaults to None.
    callback_object
        A callback object which is given to the `sel_callback` function if set.
        See also `sel_callback` in the :meth:`__init__` function. Defaults to None.

    Furthermore, all of the parameters which the :meth:`__init__` function takes are also attributes.
    """

    def __init__(self, ax, sel_callback,
                 xrange, yrange, xticks=10, yticks=10,
                 continuous_update=False,
                 draw_custom_picker=None):
        self.ax = ax
        self.sel_callback = sel_callback
        self.callback_object = None

        self.xrange, self.yrange = xrange, yrange
        self.sel_xy = (sum(xrange) / 2, sum(yrange) / 2)

        self.xticks = xticks if (isinstance(xticks, list) or isinstance(xticks, np.ndarray) or isinstance(xticks, range)) \
                             else np.linspace(*xrange, xticks)
        self.yticks = yticks if (isinstance(yticks, list) or isinstance(yticks, np.ndarray) or isinstance(yticks, range)) \
                             else np.linspace(*yrange, yticks)

        self.xlabel, self.ylabel = None, None


        self._cid = []
        self.continuous_update = continuous_update
        self.mouse_is_down = False

        self.draw_custom_picker = draw_custom_picker


    def show(self, sel_x=None, sel_y=None, with_callback=False):
        """
        Connects all of the listeners and redraws the plot. 
        
        Optionally with a default selection.

        Parameters
        ----------
        sel_x : int, optional
            The default x-coordinate. Must be given if the `with_callback` flag is set.
        sel_y : int, optional
            The default y-coordinate. Must be given if the `with_callback` flag is set.
        with_callback : bool
            If this flag is set, a default selection is made and the callback is called with the default selection.
        """

        fig = self.ax.get_figure()
        for c in self._cid:
            self.ax.get_figure().mpl_disconnect(c)
        self._cid = [fig.canvas.mpl_connect('button_press_event', self._onclick)]
        self._cid.append(fig.canvas.mpl_connect('key_press_event', self._onkey))
        if self.continuous_update:
            self._cid.append(fig.canvas.mpl_connect('motion_notify_event', self._onmove))
            self._cid.append(fig.canvas.mpl_connect('button_release_event', self._onrelease))

        self.sel_xy = sel_x, sel_y
        if with_callback:
            self._callback(sel_x, sel_y)
        self.update()


    def update(self):
        """Redraws the plot."""

        if self.draw_custom_picker:
            self.draw_custom_picker(self.ax, self.sel_xy)
        else:
            self._draw_default_picker(self.ax, self.sel_xy)
        self.ax.get_figure().canvas.draw_idle()

    def _draw_default_picker(self, ax, sel_xy):
        ax.clear()
        ax.set_ylim(*self.yrange)
        ax.set_xlim(*self.xrange)
        if sel_xy is not None:
            ax.scatter(*sel_xy, marker='o')
        ax.set_yticks(self.yticks)
        ax.set_xticks(self.xticks)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.grid(True)

    def _callback(self,x,y):
        x,y = np.clip(x, *self.xrange), np.clip(y, *self.yrange)
        if self.callback_object is None:
            self.sel_xy = self.sel_callback(x,y)
        else:
            self.sel_xy = self.sel_callback(x,y, self.callback_object)

    def _onkey(self, event):
        if event.key=="left":
            self._callback(self.sel_xy[0]-1, self.sel_xy[1])
        elif event.key=="right":
            self._callback(self.sel_xy[0]+1, self.sel_xy[1])
        elif event.key=="up":
            self._callback(self.sel_xy[0], self.sel_xy[1]+1)
        elif event.key=="down":
            self._callback(self.sel_xy[0], self.sel_xy[1]-1)
        else:
            return
        self.update()

    def _onclick(self, event):
        if event.inaxes == self.ax:
            self.mouse_is_down = self.continuous_update
            self._callback(event.xdata, event.ydata)
            self.update()

    def _onmove(self, event):
        if self.mouse_is_down and event.inaxes == self.ax:
            self._callback(event.xdata, event.ydata)
            self.update()

    def _onrelease(self, event):
        if self.mouse_is_down and event.inaxes == self.ax:
            self.mouse_is_down = False


if __name__ == "__main__":
    #import matplotlib as mpl
    #mpl.use('MacOSX')
    import numpy as np
    fig, axes = plt.subplots(nrows=2,ncols=1, figsize=(10,10))

    def cb(sel_x, sel_y, x):
        axes[0].clear()
        axes[0].set_xlim(x[0], x[-1])
        axes[0].set_ylim(-1.5,  1.5)
        axes[0].plot(x, sel_y*np.cos(x*sel_x))
        return sel_x, sel_y

    test = XYPicker(axes[1], cb, [0, 10], [0,2], xticks=range(10), yticks=[0,0.5,1,1.5])
    test.xlabel = "Frequency"
    test.ylabel = "Amplitude"
    test.callback_object = np.arange(-np.pi, np.pi, 0.01)
    test.show()

    plt.show()
