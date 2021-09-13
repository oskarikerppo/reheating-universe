#!/usr/bin/python
# -*- coding: latin-1 -*-
from __future__ import division
import numpy as np
from scipy import special
from scipy.special import hankel1 as h1
from scipy.special import hankel2 as h2
import scipy.integrate as integrate
from scipy.optimize import fsolve, minimize
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
import pickle as pickle
import sys
import logging

# Implements the numerical calulations introduced in https://arxiv.org/pdf/2010.13569.pdf

today = time.strftime("%d-%m-%Y-%H:%M:%S")
logging.basicConfig(
    filename=f"Logs/reheating-scenario-{today}.txt",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# The constant n: 1 for stiff matter, 2 for radiation, 4 for matter


# Integration settings
# Tolerances for quad
eabs = 1.49e-08
erel = 1.49e-08
lmt = 10000

# Resolution for figures
resolution = 50

################################################ HELPER FUNCTIONS ##################################################


def timing(f):
    """Timing decorator for functions."""

    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print("%s function took %0.3f ms" % (f.func_name, (time2 - time1) * 1000.0))
        return ret

    return wrap


def Gamma_phi(t, m, b):
    """Creation rate for massive particles"""
    airy_param = -((3 * m * t / 2.0) ** (2 / 3.0))
    ai, aip, bi, bip = special.airy(airy_param)
    return 3 * ((m * b) ** (13 / 3.0)) * t * (ai ** 2 + bi ** 2) / (32.0 * b)


def Gamma_chi(t, m, a, l, n):
    """Differential decay rate of massive scalars into massless conformally coupled scalars."""
    mt = m * t
    return np.real((l ** 2) * t * h1(a, mt) * h2(a, mt) / 32)


def Gamma_psi(t, m, a, h, n):
    """Differential decay rate of massive scalars into massless fermions."""
    mt = m * t
    return np.real(
        (h ** 2)
        * t
        * (h1(a - 1, mt) - h1(a + 1, mt) + ((1 - n) / t) * h1(a, mt))
        * (h2(a - 1, mt) - h2(a + 1, mt) + ((1 - n) / t) * h2(a, mt))
        / 128
    )


def alpha(n, xi):
    """Index for Hankel functions."""
    # minkowskian
    return 1 / 2.0
    # return ((1 - n*(n - 2)*(6*xi - 1))**(1/2.0))/(2.0 + n)
    # NEW
    # return (((1 - n)**2 - 4*n*(2*n - 1)*(6*xi - 1))**(1/2))/2


def bessel(t, m, a, l):
    """Definition of Bessel function."""
    return (
        ((l * t) ** 2.0)
        * (
            (special.jv(a, m * t) ** 2)
            - special.jv(a - 1, m * t) * special.jv(a + 1, m * t)
            - special.yv(a + 1, m * t) * special.yv(a - 1, m * t)
            + (special.yv(a, m * t) ** 2)
        )
        / 64.0
    )


def f(t, t_0, n, xi, m, l, h):
    """A common integrand in the article."""
    a = alpha(n, xi)
    return integrate.quad(
        lambda x: Gamma_chi(x, m, a, l, n) + Gamma_psi(x, m, a, h, n),
        t_0,
        t,
        epsabs=eabs,
        epsrel=erel,
        limit=lmt,
    )[0]


def scale_factor(t):
    """Scale factor of the universe."""
    return t ** (1 / 3.0)


def rho_phi(t, t_0, n, xi, m, l, h, b):
    """Energy density for massive particles."""
    return (
        (scale_factor(t) ** (-3.0))
        * np.exp(-f(t, t_0, n, xi, m, l, h))
        * integrate.quad(
            lambda x: (scale_factor(x) ** 3.0)
            * Gamma_phi(x, m, b)
            * np.exp(f(x, t_0, n, xi, m, l, h)),
            t_0,
            t,
            epsabs=eabs,
            epsrel=erel,
            limit=lmt,
        )[0]
    )


def rho_psi(t, t_0, n, xi, m, l, h, b):
    """Energy density for massless particles."""
    a = alpha(n, xi)
    return (scale_factor(t) ** (-4.0)) * integrate.quad(
        lambda x: (Gamma_psi(x, m, a, h, n) + Gamma_chi(x, m, a, l, n))
        * rho_phi(x, t_0, n, xi, m, l, h, b)
        * (scale_factor(x) ** 4.0),
        t_0,
        t,
        epsabs=eabs,
        epsrel=erel,
        limit=lmt,
    )[0]


def rho_stiff(t, G_N):
    """Energy density of stiff matter"""
    return 1 / (24 * np.pi * G_N * (t ** 2))


def rho_stiff_mat(t, t_0, G_N):
    """Energy density of stiff matter in matter dominated universe."""
    return (t_0 ** 4) / (24 * np.pi * G_N * (t_0 ** 2) * (t ** 4))


def rho_stiff_rad(t, t_0, t_1, G_N):
    """Energy density of stiff matter in radiation dominated universe."""
    return (
        (t_0 ** 4)
        * (t_1 ** 3)
        / (24 * np.pi * G_N * (t_0 ** 2) * (t_1 ** 4) * (t ** 3))
    )


def rho_stiff_rad_no_mat(t, t_0, G_N):
    """Energy density of stiff matter when unvierse went straight into radiation dominance."""
    return (t_0 ** 3) / (24 * np.pi * G_N * (t_0 ** 2) * (t ** 3))


################################################# THE AGORITHM #################################################

################################################# UINVERSE ENDS UP IN MATTER DOMINATED ERA #################################################


def matter_minus_stiff(t, t_0, n, xi, m, l, h, b, G_N):
    """Difference of matter and stiff energy densities."""
    return float(rho_stiff(t, G_N) - rho_phi(t, t_0, n, xi, m, l, h, b))


def rad_minus_stiff(t, t_0, n, xi, m, l, h, b, G_N):
    """Difference between radiation and stiff energy densities."""
    return float(rho_stiff(t, G_N) - rho_psi(t, t_0, n, xi, m, l, h, b))


def eq_time(t, t_0, n, xi, m, l, h, b, G_N):
    """Time when matter and stiff energy densities coincide."""
    data = (t_0, n, xi, m, l, h, b, G_N)
    t_eq = fsolve(matter_minus_stiff, t, args=data)
    return t_eq[0]


def eq_time2(t, t_0, n, xi, m, l, h, b, G_N):
    """Time when radiation and stiff energy densities coincide."""
    data = (t_0, n, xi, m, l, h, b, G_N)
    t_eq = fsolve(rad_minus_stiff, t, args=data)
    r_p = rho_psi(t_eq, t_0, n, xi, m, l, h, b)
    r_s = rho_stiff(t_eq, G_N)
    return t_eq[0]


############### MATTER DOMINATED PHASE ##################


def rho_phi_mat(t, n, xi, m, l, h, eq_time, rho_eq):
    """Energy density of matter."""
    return ((eq_time / t) ** 2.0) * (np.exp(-f(t, eq_time, n, xi, m, l, h)) * rho_eq)


def rho_psi_mat(t, n, xi, m, l, h, eq_time, rho_psi_eq, rho_eq):
    """Energy density of radiation."""
    a = alpha(n, xi)
    return (1 / (t ** (8 / 3.0))) * integrate.quad(
        lambda x: (Gamma_psi(x, m, a, h, n) + Gamma_chi(x, m, a, l, n))
        * rho_phi_mat(x, n, xi, m, l, h, eq_time, rho_eq)
        * (x ** (8 / 3.0)),
        eq_time,
        t,
        epsabs=eabs,
        epsrel=erel,
        limit=lmt,
    )[0] + rho_psi_eq * (eq_time / t) ** (8 / 3.0)


def matter_minus_rad(t, n, xi, m, l, h, eq_time, rho_psi_eq, rho_eq):
    """Time when radiation and matter energy densities coincide."""
    return rho_phi_mat(t, n, xi, m, l, h, eq_time, rho_eq) - rho_psi_mat(
        t, n, xi, m, l, h, eq_time, rho_psi_eq, rho_eq
    )


def eq_tau(t, n, xi, m, l, h, eq_time, rho_psi_eq, rho_eq):
    """Solve when radiation dominated era begins."""
    data = (n, xi, m, l, h, eq_time, rho_psi_eq, rho_eq)
    t_eq = fsolve(matter_minus_rad, t, args=data)
    return t_eq[0]


###################### Radiation dominated era #####################


def rho_phi_rad(t, n, xi, m, l, h, eq_tau, rho_tau):
    """Energy density of radiation dominated era."""
    return ((eq_tau / t) ** (3 / 2)) * (np.exp(-f(t, eq_tau, n, xi, m, l, h)) * rho_tau)


def rho_psi_rad(t, n, xi, m, l, h, eq_tau, rho_tau, psi_tau):
    """Energy density for massless particles in radiation dominated era."""
    a = alpha(n, xi)
    return ((1 / t) ** 2) * integrate.quad(
        lambda x: (Gamma_chi(x, m, a, l, n) + Gamma_psi(x, m, a, h, n))
        * rho_phi_rad(x, n, xi, m, l, h, eq_tau, rho_tau)
        * (x ** 2),
        eq_tau,
        t,
        epsabs=eabs,
        epsrel=erel,
        limit=lmt,
    )[0] + psi_tau * ((eq_tau / t) ** 2)


def d_rho_psi_rad(t, n, xi, m, l, h, eq_tau, rho_tau, psi_tau):
    """Differential of massless particle energy density."""
    a = alpha(n, xi)
    return (
        -2
        * integrate.quad(
            lambda x: (Gamma_chi(x, m, a, l, n) + Gamma_psi(x, m, a, h, n))
            * rho_phi_rad(x, n, xi, m, l, h, eq_tau, rho_tau)
            * (x ** 2),
            eq_tau,
            t,
            epsabs=eabs,
            epsrel=erel,
            limit=lmt,
        )[0]
        / (t ** 3)
        + Gamma_psi(t, m, a, h, n) * rho_phi_rad(t, n, xi, m, l, h, eq_tau, rho_tau)
        - 2 * psi_tau * (eq_tau ** 2) / (t ** 3)
    )


def r_time(t, n, xi, m, l, h, eq_tau, rho_tau, psi_tau):
    """Solve for reheating time. If differential radiation density is negative,
    the initial value is the maximum."""
    d_1 = d_rho_psi_rad(eq_tau, n, xi, m, l, h, eq_tau, rho_tau, psi_tau)
    if d_1 < 0:
        return eq_tau
    data = (n, xi, m, l, h, eq_tau, rho_tau, psi_tau)
    rt = fsolve(d_rho_psi_rad, t, data)
    return rt[0]


######################## MAIN LOGIC #####################


def reheating_time(t, t_0, m, l, h, b, xi, G_N, plot):
    """Solve the reheating time and temperature with given input parameters.

    Args:
        t (float): Time: initial guess of the reheating time. Used as startpoint for optimization.
        t_0 (float): Time (t zero): The startpoint or smalles possible time. Lower bound of the integration intervals.
        m (float): Mass. Mass of the scalar field.
        l (float): lambda. Parameter of decay.
        h (float): h. Another parameter of decay.
        b (float): b. Expansion parameter.
        xi (float): xi. Coupling constant.
        G_N (float): Gravitation constant.
        plot (boolean): Plot the results if True.

    Returns:
        list: Returns all relevant data from the simulation.
        radiation dominated era: [reheating temperature, (density, reheating time), (radiation density, time of equivalence), False]
        matter dominated era: [reheating temperature, (density, reheating time), (radiation density, transition time), (matter density, transition time), True]
        Exception: list of nan values
    """
    try:
        # First calculate time when density of radiation equals that of stiff matter
        # n equals 1 at this time
        n = 1 / 3
        # Must use function eq_time with new Gamma
        etime_rad = eq_time(t, t_0, n, xi, m, l, h, b, G_N)
        # Density of radiation at etime_rad
        phi_1 = rho_phi(etime_rad, t_0, n, xi, m, l, h, b)
        # Density of matter at etime_rad
        psi_1 = rho_psi(etime_rad, t_0, n, xi, m, l, h, b)
        # If density of radiation is greater, we go straight to radiation dominated era
        if psi_1 > phi_1:
            etime_rad = eq_time2(t, t_0, n, xi, m, l, h, b, G_N)
            phi_1 = rho_phi(etime_rad, t_0, n, xi, m, l, h, b)
            psi_1 = rho_psi(etime_rad, t_0, n, xi, m, l, h, b)
            # n eqauals 2 at this time
            n = 1 / 2
            psi_rad_init = psi_1
            phi_rad_init = phi_1
            if plot:
                plt.figure("Stiff matter dominated era")
                stiff_era = np.linspace(
                    t_0, etime_rad * 1.5, resolution
                )  # 100 linearly spaced numbers
                mat = np.array([rho_phi(z, t_0, 1, xi, m, l, h, b) for z in stiff_era])
                rad = np.array([rho_psi(z, t_0, 1, xi, m, l, h, b) for z in stiff_era])
                stiff = np.array([rho_stiff(z, G_N) for z in stiff_era])
                plt.ylim(0, max(mat[-1], rad[-1]) * 1.5)
                plt.plot(stiff_era, mat, "b-")
                plt.plot(stiff_era, rad, "y-")
                plt.plot(stiff_era, stiff, "r-")

            # Radiation dominated era begins, n equals 2
            n = 1 / 2

            # Sove for the reheating temperature
            reh_time = r_time(
                etime_rad, n, xi, m, l, h, etime_rad, phi_rad_init, psi_rad_init
            )
            max_density = rho_psi_rad(
                reh_time, n, xi, m, l, h, etime_rad, phi_rad_init, psi_rad_init
            )
            # Transfer reheating time to temperature
            if plot:
                plt.figure("Radiation dominated era")
                rad_era = np.linspace(
                    etime_rad, reh_time * 1.05, resolution
                )  # 100 linearly spaced numbers
                mat = np.array(
                    [
                        rho_phi_rad(z, n, xi, m, l, h, etime_rad, phi_rad_init)
                        for z in rad_era
                    ]
                )
                rad = np.array(
                    [
                        rho_psi_rad(
                            z, n, xi, m, l, h, etime_rad, phi_rad_init, psi_rad_init
                        )
                        for z in rad_era
                    ]
                )
                d_rad = np.array(
                    [
                        d_rho_psi_rad(
                            z, n, xi, m, l, h, etime_rad, phi_rad_init, psi_rad_init
                        )
                        for z in rad_era
                    ]
                )
                stiff = np.array(
                    [rho_stiff_rad_no_mat(z, etime_rad, G_N) for z in rad_era]
                )
                plt.ylim(0, max(rad) * 1.1)
                plt.plot(rad_era, mat, "b-")
                plt.plot(rad_era, rad, "y-")
                plt.plot(rad_era, d_rad, "g-")
                plt.plot(rad_era, stiff, "r-")
                plt.show()
            return [
                max_density ** (1 / 4),
                (max_density, reh_time),
                (psi_rad_init, etime_rad),
                False,
            ]

        else:
            # n equals 4 at this era
            # Calculate intersection of stiff and matter
            n = 1 / 3
            etime_mat = eq_time(t, t_0, n, xi, m, l, h, b, G_N)

            if plot:
                plt.figure("Stiff matter dominated era")
                stiff_era = np.linspace(
                    t_0, etime_mat, resolution
                )  # 100 linearly spaced numbers
                mat = np.array([rho_phi(z, t_0, 1, xi, m, l, h, b) for z in stiff_era])
                rad = np.array([rho_psi(z, t_0, 1, xi, m, l, h, b) for z in stiff_era])
                stiff = np.array([rho_stiff(z, G_N) for z in stiff_era])
                plt.ylim(0, mat[-1] * 1.1)
                plt.plot(stiff_era, mat, "b-")
                plt.plot(stiff_era, rad, "y-")
                plt.plot(stiff_era, stiff, "r-")

            phi_1 = rho_phi(etime_mat, t_0, n, xi, m, l, h, b)
            psi_1 = rho_psi(etime_mat, t_0, n, xi, m, l, h, b)

            # Matter dominated era begins, and n is now 4
            n = 2 / 3

            etime_rad = eq_tau(etime_mat, n, xi, m, l, h, etime_mat, psi_1, phi_1)

            phi_2 = rho_phi_mat(etime_rad, n, xi, m, l, h, etime_mat, phi_1)
            psi_2 = rho_psi_mat(etime_rad, n, xi, m, l, h, etime_mat, psi_1, phi_1)

            if plot:
                plt.figure("Matter dominated era")
                mat_era = np.linspace(
                    etime_mat, etime_rad * 1, resolution
                )  # 100 linearly spaced numbers
                mat = np.array(
                    [rho_phi_mat(z, n, xi, m, l, h, etime_mat, phi_1) for z in mat_era]
                )
                rad = np.array(
                    [
                        rho_psi_mat(z, n, xi, m, l, h, etime_mat, psi_1, phi_1)
                        for z in mat_era
                    ]
                )
                stiff = np.array([rho_stiff_mat(z, etime_mat, G_N) for z in mat_era])
                plt.ylim(0, max(phi_1, phi_2) * 1.1)
                plt.plot(mat_era, mat, "b-")
                plt.plot(mat_era, rad, "y-")
                plt.plot(mat_era, stiff, "r-")

            # Radiation dominated era begins, n equals 2
            n = 1 / 2

            # Sove for the reheating temperature
            reh_time = r_time(etime_rad, n, xi, m, l, h, etime_rad, phi_2, psi_2)
            max_density = rho_psi_rad(reh_time, n, xi, m, l, h, etime_rad, phi_2, psi_2)

            if plot:
                plt.figure("Radiation dominated era")
                rad_era = np.linspace(
                    etime_rad, reh_time * 1.05, resolution
                )  # 100 linearly spaced numbers
                mat = np.array(
                    [rho_phi_rad(z, n, xi, m, l, h, etime_rad, phi_2) for z in rad_era]
                )
                rad = np.array(
                    [
                        rho_psi_rad(z, n, xi, m, l, h, etime_rad, phi_2, psi_2)
                        for z in rad_era
                    ]
                )
                d_rad = np.array(
                    [
                        d_rho_psi_rad(z, n, xi, m, l, h, etime_rad, phi_2, psi_2)
                        for z in rad_era
                    ]
                )
                # print d_rad
                stiff = np.array(
                    [rho_stiff_rad(z, etime_mat, etime_rad, G_N) for z in rad_era]
                )
                plt.ylim(0, max(rad) * 1.1)
                plt.plot(rad_era, mat, "b-")
                plt.plot(rad_era, rad, "y-")
                plt.plot(rad_era, stiff, "r-")
                plt.plot(rad_era, d_rad, "g-")

                plt.show()
            return [
                max_density ** (1 / 4),
                (max_density, reh_time),
                (psi_2, etime_rad),
                (phi_1, etime_mat),
                True,
            ]
    except:
        return [np.nan, (np.nan, np.nan), (np.nan, np.nan), False]


def reheating_time_star(args):
    """Function that calculates reheating temperature in parallel pool"""
    return [args, reheating_time(*args)]


def generate_datapoints():
    """Generate datapoints for which reheating temperature is calculated."""
    data = []

    t_0 = 10 ** 10
    G_N = 1.0
    plot = False

    max_mass = -10
    min_mass = -12
    mass_points = np.logspace(min_mass, max_mass, 10, endpoint=True, base=10)

    min_b = -1
    max_b = 1
    b_points = np.logspace(min_b, max_b, 10, endpoint=True, base=10)

    minimal_xi = 0.0
    conformal_xi = 1.0 / 6
    other = 1.0 / 8

    h = 10 ** -14

    for m in mass_points:
        for b in b_points:
            for l in [10 ** -1, 10 ** -2, 10 ** -3]:
                for xi in [minimal_xi, conformal_xi]:
                    data.append((1.1 * t_0, t_0, m, m * l, h, b, xi, G_N, plot))
    return data


np.seterr(all="raise")

if __name__ == "__main__":
    """Calculate reheating temperatures in a multiprocessing pool."""
    logging.info("Starting run.")
    start = time.time()
    plot = False
    results = []
    p = Pool()
    logging.info("Generating data points.")
    data = generate_datapoints()
    logging.info("Calculating temperatures.")
    for i, x in enumerate(p.imap_unordered(reheating_time_star, data, 1)):
        results.append(x)
        sys.stderr.write("\rdone {0:%}".format(float(i) / len(data)))
    p.terminate()
    logging.info("Saving results.")
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)
    logging.info("Run finished.")
    sys.exit()
