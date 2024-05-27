from rabifun.system import *
from rabifun.plots import *
from rabifun.utilities import *

# %% interactive


def test_collective_mode_rabi():
    """This is couples to a linear combination of bathe modes in stead of to a single one, but the result is pretty much the same."""
    params = Params(
        η=0.01,
        Ω=1,
        δ=1 / 4,
        g_0=0.01,
        laser_detuning=0,
        ω_c=0.00,
        N=2,
        N_couplings=2,
        measurement_detuning=0,
        rwa=False,
        flat_energies=True,
    )

    # params.laser_off_time = params.lifetimes(5)
    t = time_axis(params, 15, 0.01)
    solution = solve(t, params)
    print(RuntimeParams(params))
    # signal = output_signal(t, solution.y, params)

    # f, (_, ax) = plot_simulation_result(
    #     make_figure(), t, signal, params, window=(params.lifetimes(5), t[-1])
    # )
    # # ax.axvline(params.laser_off_time)
    # plot_rabi_sidebands(ax, params)

    fig = make_figure()
    ax = fig.subplots()

    rotsol = in_rotating_frame(t, solution.y, params)

    ax.plot(t, np.abs(rotsol)[1])
    ax.plot(t, np.abs(rotsol)[2])
    ax.plot(t, np.abs(rotsol)[3])
    # ax.plot(t, np.abs(rotsol)[4])
    # ax.plot(t, np.abs(in_rotating_frame(t, solution.y, params))[2])
    # ax.plot(t, np.abs(in_rotating_frame(t, solution.y, params))[3])
