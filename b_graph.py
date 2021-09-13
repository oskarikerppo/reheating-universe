import pickle as pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from tqdm import tqdm
import time
import logging

today = time.strftime("%d-%m-%Y-%H:%M:%S")
logging.basicConfig(
    filename=f"Logs/b-graph-{today}.txt",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

matplotlib.rc("text", usetex=True)

logging.info("Opening results.")
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

# Separate datapoints based on whether the parameters led to a matter dominated phase
mat_temps = []
rad_temps = []

for x in temps:
    if x[-1]:
        mat_temps.append(x)
    else:
        rad_temps.append(x)

# Generate a grid of data points for plotting
mass_points = list(set([x[1] for x in temps]))
b_points = list(set([x[3] for x in temps]))

mass_points = sorted(mass_points)
b_points = sorted(b_points)

b_p = b_points[-1]


def create_Z(mass_points, l, xi, b):
    """Filter data points from all data based on couplings l, xi and b."""
    data = [h for h in temps if h[4] == xi and h[2] == l * h[1] and h[3] == b]
    points = [(data[k][0], data[k][1]) for k in range(len(data))]  # (temp, mass)
    points = sorted(points, key=lambda x: x[1])
    return points


for xi in [0, 1 / 6]:
    for b_p in [b_points[0], b_points[int(len(b_points) / 2)], b_points[-1]]:
        logging.info(f"Create plot: xi {xi}, b {b_p}")
        points = create_Z(mass_points, 10 ** -1, xi, b_p)
        plt.plot([p[1] for p in points], [p[0] for p in points], "b-")

        points = create_Z(mass_points, 10 ** -2, xi, b_p)
        plt.plot([p[1] for p in points], [p[0] for p in points], "g--")

        points = create_Z(mass_points, 10 ** -3, xi, b_p)
        plt.plot([p[1] for p in points], [p[0] for p in points], "r:")

        plt.xlabel("$m$", fontsize=16)
        plt.ylabel("$T_{rh}$", rotation="horizontal", fontsize=16)
        plt.xscale("log")
        plt.yscale("log")
        plt.title("")

        # Save figure
        if xi == 0:
            xi_path = "Minimal"
        elif xi == 1 / 6:
            xi_path = "Conformal"
        else:
            xi_path = "Other"
        save_path = r"Figures/{}/b_graph_{}.pdf".format(xi_path, b_p)
        plt.savefig(save_path)
        plt.close()
        # plt.show()
