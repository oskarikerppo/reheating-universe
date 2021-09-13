import pickle as pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import numpy as np
from tqdm import tqdm
import time
import copy
import logging

today = time.strftime("%d-%m-%Y-%H:%M:%S")
logging.basicConfig(
    filename=f"Logs/load-results-{today}.txt",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
matplotlib.rc("text", usetex=True)

logging.info("Load results.")
with open("results.pkl", "rb") as f:
    results = pickle.load(f)

temps = []
for x in results:
    try:
        # Matter dominated data point
        temps.append(
            [
                x[1][0],
                x[0][2],
                x[0][3],
                x[0][5],
                x[0][6],
                x[1][-1],
                x[1][1][1],
                x[1][2][1],
                x[1][3][1],
            ]
        )
    except:
        # Radiation dominated data point
        temps.append(
            [
                x[1][0],
                x[0][2],
                x[0][3],
                x[0][5],
                x[0][6],
                x[1][-1],
                x[1][1][1],
                x[1][2][1],
                np.nan,
            ]
        )


# Separate data points based on wheter data point led to matter dominated phase.
mat_temps = []
rad_temps = []

for x in temps:
    if x[-1]:
        mat_temps.append(x)
    else:
        rad_temps.append(x)

mass_points = list(set([x[1] for x in temps]))
b_points = list(set([x[3] for x in temps]))

mass_points = sorted(mass_points)

b_points = sorted(b_points)

X, Y = np.meshgrid(mass_points, b_points)


def create_string_tick(x):
    """Create numerical ticks for plots."""
    x = "{:.2e}".format(x)
    tick = ""
    dot_reached = False
    num_of_decimals = 2
    current_decimals = 0
    e_reached = False
    for c in x:
        if not dot_reached and c != ".":
            tick += c
        elif c == ".":
            tick += "."
            dot_reached = True
        elif dot_reached and current_decimals < num_of_decimals:
            tick += c
            current_decimals += 1
        elif c == "e":
            tick += c
            e_reached = True
        elif e_reached:
            tick += c
    return tick


def create_Z(x, y, l, xi):
    """Create datapoints based on couplings lambda and xi."""
    Z = np.zeros((len(x), len(x[0])))
    T = np.zeros((len(x), len(x[0])))
    T2 = np.zeros((len(x), len(x[0])))
    T3 = np.zeros((len(x), len(x[0])))
    M = np.zeros((len(x), len(x[0])))
    data = [h for h in temps if h[4] == xi and h[2] == l * h[1]]
    for i in tqdm(range(len(x))):
        for j in tqdm(range(len(x[0]))):
            for k in range(len(data)):
                if data[k][1] == x[i][j] and data[k][3] == y[i][j]:
                    z = data[k][0]
                    if z == 0:
                        Z[i][j] = np.nan
                    else:
                        Z[i][j] = z
                    if data[k][5]:
                        M[i][j] = 1
                    else:
                        M[i][j] = 0
                    T[i][j] = data[k][-3]
                    T2[i][j] = data[k][-2]
                    if data[k][-1] == 0:
                        T3[i][j] = np.nan
                    else:
                        T3[i][j] = data[k][-1]
                    data.pop(k)
                    break
    return Z, M, T, T2, T3


for lam in [10 ** -1, 10 ** -2, 10 ** -3]:
    for xi in [0, 1 / 6]:
        logging.info(f"Create plot: lambda {lam}, xi {xi}")
        Z, M, T, T2, T3 = create_Z(X, Y, lam, xi)
        fig, ax = plt.subplots()
        im = ax.pcolor(
            X,
            Y,
            Z,
            cmap="plasma",
            norm=colors.LogNorm(vmin=np.nanmin(Z), vmax=np.nanmax(Z)),
        )
        # ax.scatter(X, Y)
        ticks = np.logspace(
            np.log10(np.nanmin(Z)), np.log10(np.nanmax(Z)), 6, endpoint=True, base=10
        )
        ticks_labels = [create_string_tick(x) for x in ticks]
        cbar = fig.colorbar(im, ticks=ticks)
        cbar.ax.set_yticklabels(ticks_labels)
        fig2, ax2 = plt.subplots()
        im2 = ax2.pcolor(X, Y, M, cmap="plasma", vmin=0, vmax=1)
        # ax2.scatter(X, Y)
        ticks2 = [0, 1]
        ticks_labels2 = ["Radiation", "Matter"]
        cbar2 = fig2.colorbar(im2, ticks=ticks2)
        cbar2.ax.set_yticklabels(ticks_labels2)

        fig3, ax3 = plt.subplots()
        im3 = ax3.pcolor(
            X, Y, T, cmap="plasma", norm=colors.LogNorm(vmin=np.min(T), vmax=np.max(T))
        )
        # ax.scatter(X, Y)
        ticks3 = np.logspace(
            np.log10(np.min(T)), np.log10(np.max(T)), 6, endpoint=True, base=10
        )
        ticks_labels3 = [create_string_tick(x) for x in ticks3]
        cbar3 = fig3.colorbar(im3, ticks=ticks3)
        cbar3.ax.set_yticklabels(ticks_labels3)

        fig4, ax4 = plt.subplots()
        im4 = ax4.pcolor(
            X,
            Y,
            T2,
            cmap="plasma",
            norm=colors.LogNorm(vmin=np.min(T2), vmax=np.max(T2)),
        )
        # ax.scatter(X, Y)
        ticks4 = np.logspace(
            np.log10(np.min(T2)), np.log10(np.max(T2)), 6, endpoint=True, base=10
        )
        ticks_labels4 = [create_string_tick(x) for x in ticks4]
        cbar4 = fig4.colorbar(im4, ticks=ticks4)
        cbar4.ax.set_yticklabels(ticks_labels4)
        c_cmap = copy.copy(matplotlib.cm.plasma)
        c_cmap.set_bad("black", 1.0)
        m_array = np.ma.array(T3, mask=np.isnan(T3))
        fig5, ax5 = plt.subplots()
        im5 = ax5.pcolormesh(
            X,
            Y,
            m_array,
            cmap=c_cmap,
            norm=colors.LogNorm(vmin=np.nanmin(T3), vmax=np.nanmax(T3)),
        )
        # im5 = ax5.scatter(X, Y, c=T3, s=T3 ,cmap=c_cmap, vmin=np.nanmin(T3), vmax=np.nanmax(T3))
        # ax5.scatter(X, Y, c=T3, vmin=np.nanmin(T3), vmax=np.nanmax(T3))
        ticks5 = np.logspace(
            np.log10(np.nanmin(T3)), np.log10(np.nanmax(T3)), 6, endpoint=True, base=10
        )
        ticks_labels5 = [create_string_tick(x) for x in ticks5]
        cbar5 = fig5.colorbar(im5, ticks=ticks5)
        cbar5.ax.set_yticklabels(ticks_labels5)

        # Temperature as a function of mass and b
        ax.set_title("")
        ax.set_xlabel("$m$", fontsize=16)
        ax.set_ylabel("$b$", fontsize=16, rotation="horizontal")
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax2.set_title("")
        ax2.set_xlabel("$m$", fontsize=16)
        ax2.set_ylabel("$b$", fontsize=16, rotation="horizontal")
        ax2.set_xscale("log")
        ax2.set_yscale("log")

        # Age of universe at reheating time
        ax3.set_title("")
        ax3.set_xlabel("$m$", fontsize=16)
        ax3.set_ylabel("$b$", fontsize=16, rotation="horizontal")
        ax3.set_xscale("log")
        ax3.set_yscale("log")

        ax4.set_title("")
        ax4.set_xlabel("$m$", fontsize=16)
        ax4.set_ylabel("$b$", fontsize=16, rotation="horizontal")
        ax4.set_xscale("log")
        ax4.set_yscale("log")

        # Time of transition to matter dominance
        ax5.set_title("")
        ax5.set_xlabel("$m$", fontsize=16)
        ax5.set_ylabel("$b$", fontsize=16, rotation="horizontal")
        ax5.set_xscale("log")
        ax5.set_yscale("log")

        # Save figures
        # Save path

        if xi == 0:
            xi_path = "Minimal"
        elif xi == 1 / 6:
            xi_path = "Conformal"
        else:
            xi_path = "Other"
        if lam == 10 ** -1:
            lambda_path = "-1"
        elif lam == 10 ** -2:
            lambda_path = "-2"
        elif lam == 10 ** -3:
            lambda_path = "-3"
        elif lam == 10 ** -4:
            lambda_path = "-4"
        else:
            lambda_path = "-5"

        path = r"Figures/{}/{}".format(xi_path, lambda_path)
        fig.savefig(path + "/Figure_1.pdf")
        fig2.savefig(path + "/Figure_2.pdf")
        fig3.savefig(path + "/Figure_3.pdf")
        fig4.savefig(path + "/Figure_4.pdf")
        fig5.savefig(path + "/Figure_5.pdf")
        plt.close(fig=fig)
        plt.close(fig=fig2)
        plt.close(fig=fig3)
        plt.close(fig=fig4)
        plt.close(fig=fig5)


# plt.show()
