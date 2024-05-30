from rabifun.system import *
from rabifun.plots import *
from rabifun.utilities import *

# %% interactive


def test_collective_mode_rabi():
    """This is couples to a linear combination of bathe modes in stead of to a single one, but the result is pretty much the same."""
    ω_c = 0.1 / 2
    params = Params(
        η=0.5,
        Ω=13,
        δ=1 / 4,
        g_0=ω_c / 5,
        laser_detuning=0,
        ω_c=ω_c,
        N=2,
        N_couplings=2,
        measurement_detuning=0,
        α=0.1,
        rwa=False,
        flat_energies=True,
    )

    params.laser_off_time = params.lifetimes(10)
    # params.initial_state = make_zero_intial_state(params)
    # params.initial_state[1] = 1

    t = time_axis(params, 10, 0.01)
    # solution = solve(t, params)
    # print(RuntimeParams(params))
    # signal = output_signal(t, solution.y, params)

    # f, (_, ax) = plot_simulation_result(
    #     make_figure(), t, signal, params, window=(params.lifetimes(5), t[-1])
    # )
    # # ax.axvline(params.laser_off_time)
    # plot_rabi_sidebands(ax, params)

    fig = make_figure()
    ax = fig.subplots()

    sol_nonrwa, sol_nonrwa, *_ = plot_rwa_vs_real_amplitudes(ax, t, params)
    ax.axvline(params.laser_off_time, color="black", linestyle="--")

    print(sol_nonrwa.y[2][-1] / sol_nonrwa.y[3][-1])
    runtime = RuntimeParams(params)
    print(runtime.drive_amplitudes[0] / runtime.drive_amplitudes[1])
