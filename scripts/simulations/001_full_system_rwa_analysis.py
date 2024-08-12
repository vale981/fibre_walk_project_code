from rabifun.system import *
from rabifun.plots import *
from rabifun.utilities import *
import itertools
# %% interactive


def make_params(ω_c=0.1, N=10, gbar=1 / 3, compensate=1):
    """
    Make a set of parameters for the system with the current
    best-known settings.
    """
    return Params(
        η=0.2,
        η_hybrid=0.2,
        Ω=13,
        δ=1 / 4,
        ω_c=ω_c,
        g_0=ω_c * gbar,
        laser_detuning=0,
        N=2 * N + 2,
        N_couplings=N,
        measurement_detuning=0,
        α=0,
        rwa=False,
        flat_energies=False,
        correct_lamb_shift=compensate,
        laser_off_time=0,
    )


def decay_rwa_analysis():
    """
    Plot the population of the A site for different values of N and α,
    with and without the RWA.

    The parameters are chosen corresponding to the current best-known
    values from the experiment.

    The cutoff frequency is chosen to be 0.05Ω (which is the strongest
    that still kinda works.).
    """

    ω_c = 0.1
    Ns = [5, 10]
    gbar = 1 / 4

    fig = make_figure("decay_test", figsize=(15, len(Ns) * 3))
    ax_ns = fig.subplots(len(Ns), 2)

    have_legend = False

    results = {}
    param_dict = {}

    for i, N in enumerate(Ns):
        params = make_params(ω_c=ω_c, N=N, gbar=gbar, compensate=1)
        params.laser_off_time = 0
        params.initial_state = make_zero_intial_state(params)
        params.initial_state[1] = 1

        a_site = a_site_indices(params)[1]

        ax_real, ax_corrected = ax_ns[i]

        t = time_axis(params, recurrences=1.1, resolution=0.1)

        for α in np.linspace(0, 2, 3):
            params.α = α
            sol_nonrwa, sol_rwa = solve_nonrwa_rwa(t, params)

            results[(N, α, True)] = sol_rwa
            results[(N, α, False)] = sol_nonrwa
            param_dict[(N, α)] = params

            color = None
            for correct, rwa in itertools.product([True, False], [True, False]):
                sol = sol_rwa if rwa else sol_nonrwa
                ax = ax_corrected if correct else ax_real
                y = sol.y
                if correct:
                    y = correct_for_decay(sol, params)

                l = ax.plot(
                    sol.t,
                    np.abs(y[a_site]) ** 2,
                    label=f"{'rwa' if rwa else ''} α={α}",
                    linestyle="--" if rwa else "-",
                    alpha=0.5 if rwa else 1,
                    color=color,
                )
                color = l[0].get_color()

        ax_real.set_title(f"Real, N={N}")
        ax_corrected.set_title(f"Decay Removed, N={N}")
        for ax in [ax_real, ax_corrected]:
            if not have_legend:
                ax.legend()
                have_legend = True

            ax.set_xlabel("t [1/Ω]")
            ax.set_ylabel("Population")
            ax.set_ylim(0, 1)
            ax.set_xlim(0, t[-1])
            ax.axvline(recurrence_time(params), color="black", linestyle="--")

    fig.tight_layout()
    fig.suptitle(
        f"Decay test for η/2π={params.η}MHz, Ω/2π={params.Ω}MHz, δ/Ω={params.δ}, ω_c/Ω={params.ω_c}, g_0/ω_c={params.g_0/params.ω_c:.2f}"
    )

    save_figure(fig, "001_decay_test", extra_meta=dict(params=param_dict, Ns=Ns))
    quick_save_pickle(
        dict(results=results, params=param_dict, Ns=Ns),
        "001_decay_test",
        param_dict=param_dict,
    )


# %% make all figures
decay_rwa_analysis()
