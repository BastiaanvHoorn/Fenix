import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

import py3Dmol
import math
import pymbar

import numpy as np
import matplotlib.pyplot as plt
import veloxchem as vlx

kb: float = 1.987204259e-3  # kcal/molK
joule_to_cal: float = 0.239001


def show_snapshots(folder):
    # Read the pdb file and split it into models
    with open(f"{folder}/traj_combined.pdb", "r") as file:
        models = file.read().split("ENDMDL")

    # Extract the first and last model
    first_model = models[0] + "ENDMDL"
    last_model = models[-2] + "ENDMDL"  # -2 because the last element is an empty string

    # Display the first model
    view = py3Dmol.view(width=400, height=300)
    view.addModel(first_model, "pdb", {"keepH": True})
    view.setStyle({}, {"stick": {}, "sphere": {"scale": 0.25}})
    view.zoomTo()
    view.show()

    # Display the last model
    view = py3Dmol.view(width=400, height=300)
    view.addModel(last_model, "pdb", {"keepH": True})
    view.setStyle({}, {"stick": {}, "sphere": {"scale": 0.25}})
    view.zoomTo()
    view.show()


def beta(T: float):
    return 1 / (kb * T)


def load_energies(E_file: str):
    E = np.loadtxt(E_file)
    E1_ref = E[:, 0] * joule_to_cal
    E2_ref = E[:, 1] * joule_to_cal
    E1_run = E[:, 2] * joule_to_cal
    E2_run = E[:, 3] * joule_to_cal
    E_m = E[:, 4] * joule_to_cal
    return E1_ref, E2_ref, E1_run, E2_run, E_m


def load_lambda(lambda_file: str):
    Lambda = np.loadtxt(lambda_file)
    return Lambda


def load_ETV(ETV_file):
    ETV = np.loadtxt(ETV_file)
    step, E, T, V, Lambda = ETV[:, 0], ETV[:, 1], ETV[:, 2], ETV[:, 3], ETV[:, 4]
    return step, E, T, V, Lambda


def calculate_Eg_V_dE(E1, E2, alpha, H12, lambda_frame):
    E2_shifted = np.copy(E2) + alpha
    V = (1 - lambda_frame) * E1 + lambda_frame * E2_shifted
    dE = E1 - E2_shifted
    Eg = 0.5 * ((E1 + E2_shifted) - np.sqrt((E1 - E2_shifted) ** 2 + 4 * H12**2))
    return E2_shifted, V, dE, Eg


def bin(data, lam_i):
    binned_data = [[] for _ in range(np.max(lam_i) + 1)]
    for i, li in enumerate(lam_i):
        binned_data[li].append(data[i])
    binned_data = np.array(binned_data)
    return binned_data


def calculate_dGfep(dE, T, Lambda, Lambda_indices):
    de_lambda = bin(dE, Lambda_indices)
    dG_middle, dG_forward, dG_backward, dG_bar = [0.0], [0.0], [0.0], [0.0]
    for i, l in enumerate(Lambda[:-1]):
        delta_lambda = Lambda[i + 1] - l

        # if remove_correlation:
        #     fw, t0fw = timeseries_analysis(de_l[i])
        #     bw, t0bw = timeseries_analysis(de_l[i + 1])
        #     print(
        #         f"{t0fw} {len(fw)-len(de_l[i])}     {t0bw} {len(bw)-len(de_l[i+1])}"
        #     )
        #     fw = beta * dl * fw
        #     bw = beta * dl * bw
        # else:
        forward_energy = beta(T) * delta_lambda * de_lambda[i]
        backward_energy = beta(T) * delta_lambda * de_lambda[i + 1]
        try:
            average_forward = np.average(np.exp(forward_energy / 2))
            average_backward = np.average(np.exp(-backward_energy / 2))
            dg_middle = -1 / beta(T) * math.log(average_forward / average_backward)
            dg_forward = -1 / beta(T) * math.log(np.average(np.exp(forward_energy)))
            dg_backward = 1 / beta(T) * math.log(np.average(np.exp(-backward_energy)))
        except ValueError:
            print(
                f"ValueError encountered during FEP calculation, setting all dG for middle forward and backward to 0 for lambda {l}"
            )
            dg_middle = 0
            dg_forward = 0
            dg_backward = 0
        try:
            dg_bar = (
                -1
                / beta(T)
                * pymbar.other_estimators.bar(forward_energy, -backward_energy, False)[
                    "Delta_f"
                ]
            )
        except Exception as e:
            print(
                f"Error {e} encountered during BAR calculation, setting dG_bar to 0 for lambda {l}"
            )
            dg_bar = 0
        dG_middle.append(dG_middle[-1] + dg_middle)
        dG_forward.append(dG_forward[-1] + dg_forward)
        dG_backward.append(dG_backward[-1] + dg_backward)
        dG_bar.append(dG_bar[-1] + dg_bar)
        # dG = dGfep_all[-1, :] + [middle, manfw, manbw, bar]
        # dGfep_all = np.row_stack((dGfep_all, dG))
    return dG_bar, dG_middle, dG_forward, dG_backward


def calculate_dGevb_analytical(dGfep, Lambda, H12, Xrange):
    def R(de):
        return np.sqrt(de**2 + 4 * H12**2)

    def shift(xi):
        return -2 * H12**2 / R(xi)

    def arg(xi):
        return 0.5 * (1 + xi / R(xi))

    dGevb = shift(Xrange) + np.interp(arg(Xrange), Lambda, dGfep)
    return dGevb


def calculate_dGevb_discretised(
    dGfep, Eg, V, dE, T, Lambda, Lambda_indices, coordinate_bins
):
    N = len(Lambda)  # The amount of frames, every lambda value is a frame
    S = (
        len(coordinate_bins) + 1
    )  # The amount of bins, in between every value of X is a bin, and outside the range are two bins

    # Array for storing indices of dE compared to lambda and X, X on the first index, Lambda on the second index,
    bin_i = [[[] for x in range(N)] for x in range(S)]
    # And an array for immediatly storing Eg-V corresponding to that same index
    EgmV = [[[] for x in range(N)] for x in range(S)]

    # Assign indices
    for (
        i,
        de,
    ) in enumerate(dE):
        X_i = np.searchsorted(coordinate_bins, de)
        bin_i[X_i][Lambda_indices[i]].append(i)
        EgmV[X_i][Lambda_indices[i]].append(Eg[i] - V[i])

    count = []
    dGcor = []
    pns = []
    # For every X bin
    for X_i in range(S):
        count.append([])
        dGcor.append([])
        # For every lambda bin
        for l_i in range(N):
            count[-1].append(len(bin_i[X_i][l_i]))
            if any(bin_i[X_i][l_i]):
                dgcor = (-1 / beta(T)) * math.log(
                    np.average(np.exp(-beta(T) * np.array(EgmV[X_i][l_i])))
                )
            else:
                dgcor = 0
            dGcor[-1].append(dgcor)

        pns.append([])
        for l_i in range(N):
            # amount of samples in this lambda-X-bin divided by the total amount of samples in this X-bin
            count_sum = np.sum(np.array(count)[X_i, :])
            if count_sum == 0:
                pns[-1].append(0)
            else:
                pns[-1].append(
                    np.array(count)[X_i, l_i] / np.sum(np.array(count)[X_i, :])
                )

    dGevb = []
    for X_i in range(S):
        dGevb.append(np.sum(np.array(pns[X_i]) * (dGfep + np.array(dGcor[X_i]))))
    return dGevb, pns, dGcor


def calculate_extrema(dGevb):
    # find the maximum
    # start in the middle, expand window until maximum doesn't increase anymore
    barrier_index = len(dGevb) // 2
    barrier = dGevb[barrier_index]
    for i in range(1, barrier_index):
        new_barrier = max(barrier, dGevb[barrier_index - i], dGevb[barrier_index + i])
        if new_barrier > barrier:
            barrier = new_barrier
        else:
            barrier_index = (
                np.argmax(dGevb[barrier_index - i : barrier_index + i])
                + barrier_index
                - i
            )
            break

    reactant_min = np.min(dGevb[:barrier_index])
    dGevb = dGevb - reactant_min
    barrier = barrier - reactant_min
    product_min = np.min(dGevb[barrier_index:])
    reaction_free_energy = product_min
    return dGevb, reaction_free_energy, barrier


@staticmethod
def discard_data(discard, E1, E2, lambda_frame):
    del_indices = np.array([])
    unique_lambda_frame = np.unique(lambda_frame)
    discard_num = int(len(lambda_frame) * discard / len(unique_lambda_frame))

    for l in unique_lambda_frame:
        # all indices where lambda_frame = l
        indices = np.where(lambda_frame == l)

        new_del_indices = np.round(
            np.linspace(indices[0][0], indices[0][-1] - 1, discard_num)
        )

        del_indices = np.append(del_indices, new_del_indices)
    del_indices = np.array(del_indices, dtype=int)
    return (
        np.delete(E1, del_indices),
        np.delete(E2, del_indices),
        np.delete(lambda_frame, del_indices),
    )


def get_FEP_and_EVB(
    folders,
    alpha,
    H12,
    names=None,
    coordinate_bins=None,
    analytical=True,
    discretised=False,
    discard=0,  # interval at which to delete data, 0 for no deletion, fractional for percentage of data
):

    if coordinate_bins is None:
        coordinate_bins = np.linspace(-500, 500, 1000)
    if names is None:
        names = []
        for folder in folders:
            names.append(folder.split("/")[-1])
    Lambda = []
    results = {}

    for folder, name in zip(folders, names):
        E_file = f"{folder}/Energies.dat"
        Lambda_file = f"{folder}/Lambda.dat"
        ETV_file = f"{folder}/ETV_combined.dat"

        E1_ref, E2_ref, _, _, _ = load_energies(E_file)
        Lambda = load_lambda(Lambda_file)
        _, _, T, _, Lambda_frame = load_ETV(ETV_file)
        T = T[0]  # todo use T array instead

        if discard > 0:
            original_len = len(Lambda_frame)
            E1_ref, E2_ref, Lambda_frame = discard_data(
                discard, E1_ref, E2_ref, Lambda_frame
            )
            print(
                f"Discarding {name} data, original length was {original_len}, current length is {len(Lambda_frame)}"
            )
        Lambda_indices = [np.where(Lambda == l)[0][0] for l in Lambda_frame]
        _, V, dE, Eg = calculate_Eg_V_dE(E1_ref, E2_ref, alpha, H12, Lambda_frame)
        dGfep, _, _, _ = calculate_dGfep(
            dE, T, Lambda, Lambda_indices
        )  # todo use T array instead
        results[name] = {
            "dGfep": dGfep,
            "Lambda": Lambda,
        }
        if discretised:
            dGevb_discrete, pns, dGcor = calculate_dGevb_discretised(
                dGfep,
                Eg,
                V,
                dE,
                T,  # todo use T array instead
                Lambda,
                Lambda_indices,
                coordinate_bins,
            )
            (
                dGevb_discrete,
                reaction_free_energy_discretised,
                barrier_discretised,
            ) = calculate_extrema(dGevb_discrete)
            results[name]["discrete"] = {
                "EVB": dGevb_discrete,
                "free_energy": reaction_free_energy_discretised,
                "barrier": barrier_discretised,
                "pns": pns,
                "dGcor": dGcor,
            }

        if analytical:
            dGevb_analytical = calculate_dGevb_analytical(
                dGfep, Lambda, H12, coordinate_bins
            )
            (
                dGevb_analytical,
                reaction_free_energy_analytical,
                barrier_analytical,
            ) = calculate_extrema(dGevb_analytical)

            results[name]["analytical"] = {
                "EVB": dGevb_analytical,
                "free_energy": reaction_free_energy_analytical,
                "barrier": barrier_analytical,
            }
    return results


def print_EVB_results(results):
    print("Discrete\t Barrier \t\t Free Energy")
    for name, result in results.items():
        if "discrete" in result.keys():
            print(
                f"{name:<10} \t {result['discrete']['barrier']} \t {result['discrete']['free_energy']}"
            )

    print("\n")
    print("Analytical\t Barrier \t\t Free Energy")
    for name, result in results.items():
        if "analytical" in result.keys():
            print(
                f"{name:<10} \t {result['analytical']['barrier']} \t {result['analytical']['free_energy']}"
            )


def plot_EVB(results, coordinate_bins=None, plot_analytical=True, plot_discrete=True):
    if coordinate_bins is None:
        coordinate_bins = np.linspace(-500, 500, 1000)
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    bin_indicators = (coordinate_bins[:-1] + coordinate_bins[1:]) / 2
    colors = mcolors.TABLEAU_COLORS
    colorkeys = list(colors.keys())
    legend_lines = []
    legend_labels = []
    if plot_analytical and plot_discrete:
        discrete_linestyle = "--"
    else:
        discrete_linestyle = "-"
    for i, (name, result) in enumerate(results.items()):
        ax[0].plot(result["Lambda"], result["dGfep"], label=name)
        if plot_discrete:
            if "discrete" in result.keys():
                ax[1].plot(
                    bin_indicators,
                    result["discrete"]["EVB"][1:-1],
                    label=f"{name} discretised",
                    color=colors[colorkeys[i]],
                    linestyle=discrete_linestyle,
                )
        if plot_analytical:
            if "analytical" in result.keys():
                ax[1].plot(
                    bin_indicators,
                    result["analytical"]["EVB"][1:],
                    label=f"{name} analytical",
                    color=colors[colorkeys[i]],
                )

        legend_lines.append(Line2D([0], [0], color=colors[colorkeys[i]]))
        legend_labels.append(name)

    if plot_analytical and plot_discrete:
        evb_legend_lines = []
        evb_legend_labels = []
        evb_legend_lines.append(Line2D([0], [0], linestyle="-", color="grey"))
        evb_legend_labels.append("analytical")
        evb_legend_lines.append(Line2D([0], [0], linestyle="--", color="grey"))
        evb_legend_labels.append("discrete")
        ax[1].legend(evb_legend_lines, evb_legend_labels)

    # ax[0].legend()
    ax[0].set_xlabel(r"$\lambda$")
    ax[0].set_ylabel(r"$\Delta G_{FEP}$ (kcal/mol)")

    ax[1].set_xlabel(r"$\Delta \mathcal{E}$ (kcal/mol)")
    ax[1].set_ylabel(r"$\Delta G_{EVB}$ (kcal/mol)")
    fig.legend(
        legend_lines,
        legend_labels,
        loc=(0.22, 0.91),
        ncol=len(legend_labels),
    )


def fit_EVB_parameters(
    T,
    reference,
    free_energy,
    barrier,
    coordinate_bins=None,
    alpha_guess=0,
    H12_guess=20,
    tol=0.0001,
    max_iter=50,
    fitting_slowness=1,
    verbose=True,
):
    if coordinate_bins is None:
        coordinate_bins = np.linspace(-500, 500, 1000)
    E_file = f"{reference}/Energies.dat"
    Lambda_file = f"{reference}/Lambda.dat"
    ETV_file = f"{reference}/ETV_combined.dat"
    E1_ref, E2_ref, _, _, _ = load_energies(E_file)
    Lambda = load_lambda(Lambda_file)
    _, _, _, _, Lambda_frame = load_ETV(ETV_file)
    Lambda_indices = [np.where(Lambda == l)[0][0] for l in Lambda_frame]
    fitting = True
    alpha = alpha_guess
    H12 = H12_guess
    i = 0
    while fitting and i < max_iter:
        _, _, dE, _ = calculate_Eg_V_dE(E1_ref, E2_ref, alpha, H12, Lambda_frame)
        dGfep, _, _, _ = calculate_dGfep(
            dE, T, Lambda, Lambda_indices
        )  # todo use T array instead
        dGevb = calculate_dGevb_analytical(dGfep, Lambda, H12, coordinate_bins)
        dGevb, free_energy_measured, barrier_measured = calculate_extrema(dGevb)
        # if within tollerance, set fitting to False, otherwise adjust parameters
        barrier_dif = barrier - barrier_measured
        free_energy_dif = free_energy - free_energy_measured
        if abs(barrier_dif) < tol and abs(free_energy_dif) < tol:
            fitting = False
        else:
            if abs(barrier_dif) > tol:
                # increase H12 if barrier is too high, otherwise reduce H12
                H12 += barrier_dif * fitting_slowness
            if abs(free_energy_dif) > tol:
                # increase alpha if free energy is too low, otherwise reduce alpha
                alpha += free_energy_dif * fitting_slowness
        i += 1
        if verbose:
            print(
                f"iteration {i} \t alpha: {alpha:<.5f} \t H12: {H12:<.5f} \t \t barrier: {barrier_measured:<.5f} \t free energy: {free_energy_measured:<.5f}"
            )
    if i == max_iter:
        print(
            "Max iterations reached, fitting did not converge. Maybe try with more itterations, a higher tolerance or a lower fitting slowness."
        )
    return alpha, H12


def plot_energies(folder):
    E_file = f"{folder}/Energies.dat"
    Lambda_file = f"{folder}/Lambda.dat"
    ETV_file = f"{folder}/ETV_combined.dat"

    E1_ref, E2_ref, E1_run, E2_run, E_m = load_energies(E_file)
    Lambda = load_lambda(Lambda_file)
    steps, E, T, V, lambda_frame = load_ETV(ETV_file)

    _, V_ref, dE_ref, Eg_ref = calculate_Eg_V_dE(E1_ref, E2_ref, 0, 0, lambda_frame)
    _, V_run, dE_run, Eg_run = calculate_Eg_V_dE(E1_run, E2_run, 0, 0, lambda_frame)

    E1f_ref_file = f"{folder}/E1f_ref.dat"
    E2f_ref_file = f"{folder}/E2f_ref.dat"
    E1f_run_file = f"{folder}/E1f_run.dat"
    E2f_run_file = f"{folder}/E2f_run.dat"
    Vf_ref, ref_headers = get_Vf(E1f_ref_file, E2f_ref_file, lambda_frame)
    Vf_run, run_headers = get_Vf(E1f_run_file, E2f_run_file, lambda_frame)

    Efm_file = f"{folder}/Efm.dat"
    Efm = np.loadtxt(Efm_file)

    # fig = plt.figure(layout="constrained", figsize=(10, 10))
    # space = 0.1
    # subfigs = fig.subfigures(2, 2, wspace=space, hspace=space)

    # ETV_ax = plot_ETV(subfigs[0, 0], E, T, V, steps)
    # joule_to_cal: float = 0.239001
    # V_ax = plot_V(subfigs[1, 0], E * joule_to_cal, E_m, V_ref, V_run, lambda_frame)
    # diabats_ax = plot_diabats(
    #     subfigs[0, 1],
    #     dE_ref,
    #     E1_ref,
    #     E2_ref,
    #     Eg_ref,
    #     dE_run,
    #     E1_run,
    #     E2_run,
    #     Eg_run,
    # )
    # fc_ax = plot_force_contributions(
    #     subfigs[1, 1], Efm, Vf_ref, Vf_run, lambda_frame, ref_headers
    # )

    fig = plt.figure(layout="constrained", figsize=(8, 4))

    diabats_ax = plot_diabats(
        fig,
        dE_ref,
        E1_ref,
        E2_ref,
        Eg_ref,
        dE_run,
        E1_run,
        E2_run,
        Eg_run,
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Scatter",
            markerfacecolor=mcolors.TABLEAU_COLORS["tab:blue"],
            markersize=5,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Scatter",
            markerfacecolor=mcolors.TABLEAU_COLORS["tab:orange"],
            markersize=5,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Scatter",
            markerfacecolor=mcolors.TABLEAU_COLORS["tab:green"],
            markersize=5,
        ),
    ]
    labels = [r"$\mathcal{E}_1$", r"$\mathcal{E}_2$", r"$E_g$"]
    fig.legend(handles, labels, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.1))
    return fig


def plot_ETV(fig, E, T, V, steps):
    ax = fig.subplots(3, 1, sharex=True)
    # Plot E
    ax[0].plot(steps, E)
    ax[0].set_ylabel("E")

    # Plot T
    ax[1].plot(steps, T)
    ax[1].set_ylabel("T")

    # Plot V
    ax[2].plot(steps, V)
    ax[2].set_xlabel("Steps")
    ax[2].set_ylabel("V")
    return ax


def average_per_lambda(lambda_frame, quant):
    average = []
    unique_lambda_frame = np.unique(lambda_frame)
    for lam in unique_lambda_frame:
        average.append(np.mean(quant[lambda_frame == lam]))
    return np.array(average), unique_lambda_frame


def plot_V(fig, E, Em, V_ref, V_run, lambda_frame):
    ax = fig.subplots(1, 1)
    E_average, Lambda = average_per_lambda(lambda_frame, E)
    Em_average, _ = average_per_lambda(lambda_frame, Em)
    V_ref_average, _ = average_per_lambda(lambda_frame, V_ref)
    V_run_average, _ = average_per_lambda(lambda_frame, V_run)
    # ax.plot(Lambda, E_average, label="E", linewidth=3)
    ax.plot(Lambda, E_average - Em_average, label="Em", linewidth=2)
    ax.plot(Lambda, E_average - V_ref_average, label="V_ref", linewidth=1)
    ax.plot(Lambda, E_average - V_run_average, label="V_run", linewidth=1)

    # ax.plot(steps, E, linewidth=2, label="E")
    # opacity = 0.8
    # ax.plot(steps, Em, linewidth=1, alpha=opacity, label="Em")
    # ax.plot(steps, V_run, linewidth=0.5, alpha=opacity, label="V_run")
    # ax.plot(steps, V_ref, linewidth=0.5, alpha=opacity, label="V_ref")
    ax.legend()
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Energy (kcal/mol)")
    return ax


def plot_diabats(fig, dE_ref, E1_ref, E2_ref, Eg_ref, dE_run, E1_run, E2_run, Eg_run):
    ax = fig.subplots(1, 3)
    dotsize = 0.1
    ax[0].scatter(dE_run, E1_run, s=dotsize * 10)
    ax[0].scatter(dE_run, E2_run, s=dotsize * 10)
    ax[0].scatter(dE_run, Eg_run, s=dotsize, alpha=0.8)
    ax[0].set_ylabel("E (kcal/mol)")
    ax[0].set_title(r"$V_{sample}$")
    ax[0].set_xlabel(r"$\Delta \mathcal{E}$ (kcal/mol)")
    ax[1].scatter(dE_run, E1_run, s=dotsize * 10)
    ax[1].scatter(dE_run, E2_run, s=dotsize * 10)
    ax[1].scatter(dE_run, Eg_run, s=dotsize, alpha=0.8)
    ax[1].set_ylabel("E (kcal/mol)")
    ax[1].set_title(r"$V_{sample}$")
    ax[1].set_xlabel(r"$\Delta \mathcal{E}$ (kcal/mol)")
    ax[1].set_xlim(-300, 300)
    ax[1].set_ylim(-20, 300)
    ax[2].scatter(dE_ref, E1_ref, s=dotsize * 10)
    ax[2].scatter(dE_ref, E2_ref, s=dotsize * 10)
    ax[2].scatter(dE_ref, Eg_ref, s=dotsize, alpha=0.8)
    ax[2].set_xlabel(r"$\Delta \mathcal{E}$ (kcal/mol)")
    ax[2].set_ylabel("E (kcal/mol)")
    ax[2].set_title(r"$V_{recalc}$")
    ax[2].set_xlim(-300, 300)
    ax[2].set_ylim(-20, 300)
    return ax


def get_Vf(E1f_file, E2f_file, lambda_frame):
    E1f = np.loadtxt(E1f_file, skiprows=1)
    E2f = np.loadtxt(E2f_file, skiprows=1)
    with open(E1f_file, "r") as file:
        headers = file.readline().split(",")
    Lambda_frame_tiled = np.tile(lambda_frame, (len(headers), 1)).transpose()
    Vf = (1 - Lambda_frame_tiled) * E1f + Lambda_frame_tiled * E2f
    return Vf, headers


def plot_force_contributions(fig, Efm, Vf_ref, Vf_run, lambda_frame, headers):
    ax = fig.subplots(1, 1)
    tol = 0.1
    opacity = 1
    for i, force_name in enumerate(headers):
        ref_dif = Efm[:, i] - Vf_run[:, i]
        ref_dif_avg, Lambda = average_per_lambda(lambda_frame, ref_dif)
        run_dif = Efm[:, i] - Vf_ref[:, i]
        run_dif_avg, Lambda = average_per_lambda(lambda_frame, run_dif)

        if np.max(np.abs(ref_dif_avg)) > tol:
            ax.plot(Lambda, ref_dif_avg, label=f"Ref {force_name}", alpha=opacity)
            opacity = 0.7
        if np.max(np.abs(run_dif_avg)) > tol:
            ax.plot(Lambda, run_dif_avg, label=f"Run {force_name}", alpha=opacity)
            opacity = 0.7
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Energy (kcal/mol)")
    ax.legend()
    return ax
