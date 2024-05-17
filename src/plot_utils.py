import matplotlib.pyplot as plt


def wrap_plot(f):
    def wrapped(*args, ax=None, setup_function=plt.subplots, **kwargs):
        fig = None
        if not ax:
            fig, ax = setup_function()

        ret_val = f(*args, ax=ax, **kwargs)
        return (fig, ax, ret_val) if ret_val else (fig, ax)

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
