import matplotlib.pyplot as plt
from typing import Callable, Any
from functools import wraps
from typing_extensions import ParamSpec, TypeVar, Concatenate

P = ParamSpec("P")
R = TypeVar("R")


def make_figure(fig_name: str = "interactive", *args, **kwargs):
    fig = plt.figure(fig_name, *args, **kwargs)
    fig.clf()
    return fig


def wrap_plot(
    plot_function: Callable[Concatenate[plt.Figure | None, P], R],  # pyright: ignore [reportPrivateImportUsage]
) -> Callable[Concatenate[plt.Figure | None, P], tuple[plt.Figure, R]]:  # pyright: ignore [reportPrivateImportUsage]
    """Decorator to wrap a plot function to inject the correct figure
    for interactive use.  The function that this decorator wraps
    should accept the figure as first argument.

    :param fig_name: Name of the figure to create.  By default it is
        "interactive", so that one plot window will be reused.
    :param setup_function: Function that returns a figure to use.  If
        it is provided, the ``fig_name`` will be ignored.
    """

    def wrapped(fig, *args: P.args, **kwargs: P.kwargs):
        if fig is None:
            fig = make_figure()

        ret_val = plot_function(fig, *args, **kwargs)
        return (fig, ret_val)

    return wrapped


def autoclose(f):
    def wrapped(*args, **kwargs):
        plt.close()
        return f(*args, **kwargs)

    return wrapped


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
