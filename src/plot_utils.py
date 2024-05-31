import matplotlib.pyplot as plt
from typing import Callable, Any
from functools import wraps
from typing_extensions import ParamSpec, TypeVar, Concatenate
import yaml
import inspect
import subprocess
import pathlib
import sys

P = ParamSpec("P")
R = TypeVar("R")


def noop_if_interactive(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if hasattr(sys, "ps1"):
            return

        return f(*args, **kwargs)

    return wrapped


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


def get_jj_info(type):
    return subprocess.run(
        ["jj", "log", "-T", type, "-l", "1", "--no-graph"],
        stdout=subprocess.PIPE,
    ).stdout.decode("utf-8")


def write_meta(path, **kwargs):
    """Write metatdata for result that has been written to a file
    under ``path``.

    The metadata includes the change_id, commit_id and the description
    of the current ``jj`` state, and the source file that generated
    the result. Additional metadata can be provided through the
    keyword arguments.
    """
    change_id = get_jj_info("change_id")
    commit_id = get_jj_info("commit_id")
    description = get_jj_info("description")
    project_dir = (
        subprocess.run("git rev-parse --show-toplevel", shell=True, capture_output=True)
        .stdout.decode("utf-8")
        .strip()
    )

    frame = inspect.stack()[3]
    module = inspect.getmodule(frame[0])
    filename = str(
        pathlib.Path(module.__file__).relative_to(project_dir)  # type: ignore
        if module
        else "<unknown>"
    )
    function = frame.function

    outpath = f"{path}.meta.yaml"
    with open(outpath, "w") as f:
        yaml.dump(
            dict(
                source=filename,
                function=function,
                change_id=change_id,
                commit_id=commit_id,
                description=description.strip(),
                refers_to=str(path),
            )
            | kwargs,
            f,
        )

    print(f"Metadata written to {outpath}")


@noop_if_interactive
def save_figure(fig, name, extra_meta=None, *args, **kwargs):
    dir = pathlib.Path(f"./figs/")
    dir.mkdir(exist_ok=True)
    fig.tight_layout()

    write_meta(f"./figs/{name}.pdf", name=name, extra_meta=extra_meta)

    plt.savefig(f"./figs/{name}.pdf", *args, **kwargs)
    plt.savefig(f"./figs/{name}.png", *args, dpi=600, **kwargs)

    print(f"Figure saved as ./figs/{name}.pdf")


@noop_if_interactive
def quick_save_pickle(obj, name, **kwargs):
    """Quickly save an object to a pickle file with metadata."""
    import pickle

    path = pathlib.Path(f"./outputs/{name}.pkl")
    path.parent.mkdir(exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)

    write_meta(path, **kwargs)


def quick_load_pickle(name):
    """Quickly load an object from a pickle file."""

    import pickle

    path = pathlib.Path(f"./outputs/{name}.pkl")

    with open(path, "rb") as f:
        return pickle.load(f)
