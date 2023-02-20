#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Python plotting script; written by Julia Bodensteiner,
# extended by Tomer Shenar
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from .file_handler import read_file

__all__ = [
    "plot_k1k2",
    "plot_input_spectrum",
    "plot_fits",
    "plot_extremes",
    "plot_conv",
    "plot_itr",
]

# For plotting 1-D and 2-D chi2 maps and compute K1+K2 + error estimate.

###########################
######### USER INPUT ######
###########################

# File to be read
fileK1K2 = "Output/HeI4472_grid_dis_K1K2.txt"
# fileK1K2 = 'HeII4546_grid_dis_K1K2.txt'
# fileK1K2 = 'HeII4200_grid_dis_K1K2.txt'
# fileK1K2 = 'HeI4388_grid_dis_K1K2.txt'
# fileK1K2 = 'HeI4026_grid_dis_K1K2.txt'
# fileK1K2 = 'HeI4009_grid_dis_K1K2.txt'
# fileK1K2 = 'HeI4144_grid_dis_K1K2.txt'
# fileK1K2 = 'Hdelta_grid_dis_K1K2.txt'
# fileK1K2 = 'Indiv_grid_dis_K1K2.txt'

# How many indices next to minimum should parabolas be fit
step1 = 3
step2 = 3

# Cut part of the chi2 matrix?
indK1cut1, indK1cut2 = 0, -1
indK2cut1, indK2cut2 = 0, -1

# matplotlib config

# params = {'legend.fontsize': 'x-large',
#'axes.labelsize': 'x-large',
#'axes.titlesize':'large',
#'xtick.labelsize':'x-large',
#'ytick.labelsize':'x-large'}
# pylab.rcParams.update(params)
legsize = 9
alphaleg = 0.0
locleg = "lower left"


###########################
#######END USER INPUT #####
###########################


def plot_k1k2(
    fileK1K2,
    step1,
    step2,
    indK1cut1,
    indK1cut2,
    indK2cut1,
    indK2cut2,
    display=False,
):
    Z = np.loadtxt(fileK1K2)

    # MinInds =  np.where(Z == np.min(Z))

    try:
        fileK1 = fileK1K2[:-8] + "K1.txt"
        sigma1 = float((open(fileK1).readlines())[0].split()[-1])
        K1s, chis1 = np.loadtxt(fileK1, unpack=True)
        minind1 = np.argmin(chis1)
    except:
        sigma1 = 0
        K1s = [0]
        pass
    try:
        fileK2 = fileK1K2[:-8] + "K2.txt"
        sigma2 = float((open(fileK2).readlines())[0].split()[-1])
        K2s, chis2 = np.loadtxt(fileK2, unpack=True)
        minind2 = np.argmin(chis2)
    except:
        K2s = [0]
        sigma2 = 0
        pass
    sigma = (sigma1 + sigma2) / 2.0

    if len(K1s) > 1:
        a, b, c = np.polyfit(
            K1s[max(minind1 - step1, 0) : min(minind1 + step1, len(K1s) - 1)],
            chis1[
                max(minind1 - step1, 0) : min(minind1 + step1, len(K1s) - 1)
            ],
            2,
        )
        Chi2K1min = -b / 2.0 / a

        K1fine = np.arange(-K1s[-1], K1s[-1] + 1, 0.01)

        parb = a * K1fine**2 + b * K1fine + c

        LeftPart = K1fine < Chi2K1min
        K1err = Chi2K1min - K1fine[np.argmin((parb[LeftPart] - sigma1) ** 2)]

        plt.scatter(K1s, chis1)
        plt.plot(K1fine, parb)
        plt.plot(
            [min(K1s), max(K1s)],
            [sigma1, sigma1],
            color="red",
            label="1-sigma",
        )
        plt.ylabel("reduced chi2")
        plt.xlabel("K1 [km/s]")
        plt.xlim(K1s[0] - 1, K1s[-1] + 1)
        plt.ylim(min(chis1) * 0.95, max(chis1) * 1.05)
        plt.text(
            np.average(K1s),
            max(chis1),
            r"$K_1 =$ "
            + str(round(Chi2K1min, 1))
            + r"$\pm$"
            + str(round(K1err, 1))
            + " km/s",
            size=13,
            horizontalalignment="center",
        )
        plt.show()
        print("K1: ", Chi2K1min, " +- ", K1err)

    if len(K2s) > 1:
        a, b, c = np.polyfit(
            K2s[max(minind2 - step2, 0) : min(minind2 + step2, len(K2s) - 1)],
            chis2[
                max(minind2 - step2, 0) : min(minind2 + step2, len(K2s) - 1)
            ],
            2,
        )
        Chi2K2min = -b / 2.0 / a

        K2fine = np.arange(-K2s[-1], K2s[-1] + 1, 0.01)

        parb = a * K2fine**2 + b * K2fine + c

        LeftPart = K2fine < Chi2K2min
        K2err = Chi2K2min - K2fine[np.argmin((parb[LeftPart] - sigma2) ** 2)]

        plt.scatter(K2s, chis2)
        plt.plot(K2fine, parb)
        plt.plot(
            [min(K2s), max(K2s)],
            [sigma2, sigma2],
            color="red",
            label="1-sigma",
        )
        plt.ylabel("reduced chi2")
        plt.xlabel("K2 [km/s]")
        plt.xlim(K2s[0] - 1, K2s[-1] + 1)
        plt.ylim(min(chis2) * 0.95, max(chis2) * 1.05)
        plt.text(
            np.average(K2s),
            max(chis2),
            r"$K_2 =$ "
            + str(round(Chi2K2min, 1))
            + r"$\pm$"
            + str(round(K2err, 1))
            + " km/s",
            size=13,
            horizontalalignment="center",
        )
        plt.show()
        print("K2: ", Chi2K2min, " +- ", K2err)

    ###########################
    ####### 2D stuff      #####
    ###########################

    if len(K1s) > 1 and len(K2s) > 1:
        # head = open(fileK1K2).readlines()[0].split()

        MinChis1, MaxChis1, MinChis2, MaxChis2 = (
            np.amin(chis1),
            np.amax(chis1),
            np.amin(chis2),
            np.amax(chis2),
        )

        LimMin = min(MinChis1, MinChis2)
        LimMax = max(MaxChis1, MaxChis2)

        Chi2K1min, Chi2K2min = K1s[minind1], K2s[minind2]

        K1min = minind1

        fig, ax = plt.subplots()
        plt.scatter(Chi2K2min, Chi2K1min, color="green")
        X, Y = np.meshgrid(K2s[indK2cut1:indK2cut2], K1s[indK1cut1:indK1cut2])

        cs = plt.contourf(
            X,
            Y,
            Z[indK2cut1:indK2cut2, indK1cut1:indK1cut2],
            200,
            cmap="inferno_r",
            vmin=LimMin,
            vmax=LimMax,
            zorder=-1,
        )

        cont = ax.contour(
            X,
            Y,
            Z[indK2cut1:indK2cut2, indK1cut1:indK1cut2],
            [sigma],
            linewidths=2,
            linestyles="dashed",
            colors="green",
        )

        vertices = cont.collections[0].get_paths()[0].vertices
        K2min, K2max = np.amin(vertices[:, 0]), np.amax(vertices[:, 0])
        K1min, K1max = np.amin(vertices[:, 1]), np.amax(vertices[:, 1])

        plt.xlabel(r"$K_2 [{\rm km}\,{\rm s}^{-1}]$")
        plt.ylabel(r"$K_1 [{\rm km}\,{\rm s}^{-1}]$")
        cbar = plt.colorbar()
        tickchi = np.arange(0.6, 1.0, 0.01)
        cbar.ax.set_ylabel(r"$\chi_{\rm reduced}^2$", size=12)
        plt.title(fileK1K2)
        plt.savefig(fileK1K2[:-4] + "_2Dmap.pdf", bbox_inches="tight")
        if display:
            plt.show()


def plot_input_spectrum():
    # Type --legend for legend
    # Type --scatterr for scatter plot + errors
    # Other options exist, see script below...
    fig, ax = plt.subplots()

    Legend = False
    Binning = False
    ScatterErr = False
    Skip = False
    SkipTwice = False
    Norm = False
    for i in range(len(sys.argv) - 1):
        print(i)
        if Skip:
            if SkipTwice:
                Skip = True
                SkipTwice = False
            else:
                Skip = False
            continue
        j = i + 1
        if len(sys.argv) > 1:
            if sys.argv[j] == "--legend":
                Legend = True
                continue
            if sys.argv[j] == "--norm":
                Norm = True
                continue
            if sys.argv[j] == "--scatterr":
                ScatterErr = True
                continue
            if sys.argv[j] == "--cols":
                col0 = float(sys.argv[j + 1])
                col1 = float(sys.argv[j + 2])
                Skip = True
                SkipTwice = True
                continue
        infile = sys.argv[j]
        try:
            try:
                wave_in, flux_in = read_file(infile, col0, col1)
            except:
                wave_in, flux_in = read_file(infile)
            flux_in = np.nan_to_num(flux_in, 1.0)
        except:
            continue
        try:
            wave_in = wave_in.astype(float)
            flux_in = flux_in.astype(float)
        except:
            wave_in, flux_in = np.loadtxt(infile, unpack=True)
        if len(flux_in) == 2:
            flux = flux_in[0]
            err = flux_in[1]
        else:
            flux = flux_in
        if Norm:
            flux /= np.mean(flux)
        # Do the plotting
        name = str(infile).split(".fits")[0]
        if ScatterErr:
            ax.errorbar(
                wave_in,
                flux,
                yerr=np.loadtxt(infile)[:, 2],
                fmt="o",
                linewidth=1.0,
                alpha=0.8,
                label=name,
            )
        else:
            ax.plot(wave_in, flux, linewidth=1.0, alpha=0.8, label=name)
    if Legend:
        ax.legend()
        ax.set_xlabel("Wavelength [A]")
        ax.set_ylabel("Flux")
    if display:
        plt.show()


def plot_chi2(
    Ks_comp,
    redchi2,
    chi2,
    nu,
    chi2fine,
    parb,
    chi2P1,
    K_label,
    output_path,
    filename,
    display,
):
    plt.figure(figsize=(6, 6))
    plt.clf()
    plt.scatter(Ks_comp, redchi2, s=2)
    plt.scatter(Ks_comp, chi2 / nu, s=2)
    plt.plot(chi2fine, parb, color="C1")
    plt.plot(
        [Ks_comp[0], Ks_comp[-1]],
        [chi2P1, chi2P1],
        color="C3",
        label=r"1$\sigma$ contour",
    )
    plt.ylabel(r"Normalised reduced $\chi^2$")
    plt.xlabel(f"${{{K_label}}}$ [km/s]")
    plt.legend()
    plt.savefig(
        os.path.join(output_path, filename),
        bbox_inches="tight",
    )

    if display:
        plt.show()


def plot_best_fit(
    waves,
    spec_sum,
    A_shift,
    B_shift,
    neb_shift,
    obs_spec,
    label_name,
    output_path,
    star_name,
    display=False,
):
    plt.figure(figsize=(12, 8))
    plt.clf()
    plt.plot(waves, spec_sum, label="sum")
    plt.plot(waves, A_shift, label="A")
    plt.plot(waves, B_shift, label="B")
    if (neb_shift > 0).any():
        plt.plot(waves, neb_shift, label="Neb")
    plt.plot(
        waves,
        obs_spec,
        label=label_name,
    )
    plt.legend()
    file_name = f"{star_name}_disentangled_spectra.pdf"
    plt.savefig(os.path.join(output_path, file_name), bbox_inches="tight")
    if display:
        plt.show()


def plot_extremes(
    waves,
    A_shift,
    B_shift,
    neb_shift,
    obs_spec,
    spec_sum,
    spec_name,
    phis,
    mjds,
    plt_ext_ymin,
    plt_ext_ymax,
    output_path,
    star_name,
    range_str,
    K1now,
    K2now,
    neb_lines=False,
    extremes_fig_size=(8, 8),
    line_wid_ext=2,
    display=False,
):
    fig_extremes, axes = plt.subplots(
        nrows=2, ncols=1, figsize=extremes_fig_size
    )

    for panel in range(2):
        axes[panel].plot(
            waves,
            A_shift[panel],
            label="Prim. Dis.",
            color="red",
            linestyle="dotted",
            linewidth=line_wid_ext,
        )
        axes[panel].plot(
            waves,
            B_shift[panel],
            label="Sec. Dis.",
            color="green",
            linewidth=line_wid_ext,
        )
        if neb_lines:
            axes[panel].plot(
                waves, neb_shift[panel], label="Nebular Dis.", color="purple"
            )
        axes[panel].plot(
            waves,
            obs_spec[panel],
            color="blue",
            label=str(round(mjds[panel], 0))
            + r", $\varphi=$"
            + str(round(phis[panel], 2)),
        )
        axes[panel].plot(
            waves,
            spec_sum[panel],
            label="Sum Dis.",
            color="black",
            linestyle="--",
            linewidth=line_wid_ext,
        )
        axes[panel].set_title(spec_name[panel].split(os.path.sep)[-1])
        axes[panel].legend(
            prop={"size": legsize}, loc=locleg, framealpha=alphaleg
        )
        axes[panel].set_ylabel("Normalised flux")
        axes[panel].set_ylim(plt_ext_ymin[panel], plt_ext_ymax[panel])
        diff_major = int((waves[-1] - waves[0]) / 3)
        axes[panel].xaxis.set_major_locator(MultipleLocator(diff_major))
        axes[panel].xaxis.set_minor_locator(MultipleLocator(diff_major / 5.0))
        axes[panel].yaxis.set_minor_locator(MultipleLocator(0.01))

    file_name = f"{star_name}_{range_str}_Extremes_"
    file_name += f"{np.round(K1now[panel])}_{np.round(K2now[panel])}.pdf"
    plt.savefig(os.path.join(output_path, file_name), bbox_inches="tight")

    if display:
        plt.show()


def plot_convergence(eps_itr, output_path, star_name, display=False):
    plt.figure()
    plt.clf()
    plt.scatter(np.arange(len(eps_itr)), eps_itr, color="C0", s=2)
    plt.ylabel("log(Eps)")
    plt.xlabel("iteration number")
    plt.tight_layout()
    file_name = f"{star_name}_convergence.pdf"
    plt.savefig(os.path.join(output_path, file_name), bbox_inches="tight")
    if display:
        plt.show()


def plot_iteration(
    itr, waves, A, B, neb_spec, output_path, star_name, display=False
):
    plt.figure()
    plt.clf()
    plt.plot(waves, A, label=itr, lw=1)
    plt.plot(waves, B, label=itr, lw=1)
    if (neb_spec > 0).any():
        plt.plot(waves, neb_spec, label=itr)
    plt.ylabel("Normalised flux")
    plt.xlabel("Wavelength")
    plt.tight_layout()
    plt.legend()
    file_name = f"{star_name}_iteration_{itr}.pdf"
    plt.savefig(os.path.join(output_path, file_name), bbox_inches="tight")
    if display:
        plt.show()
