#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Python plotting script; written by Julia Bodensteiner,
# extended by Tomer Shenar
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

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
        plt.show()
        # np.savetxt(infile + '.txt', np.c_[wave_in, flux])


def plot_fits(
    waves,
    specsum,
    Ashift,
    Bshift,
    Nebshift,
    Obsspec,
    label_name,
    display=False,
):
    plt.plot(waves, specsum, label="sum")
    plt.plot(waves, Ashift, label="A")
    plt.plot(waves, Bshift, label="B")
    if (Nebshift > 0).any():
        plt.plot(waves, Nebshift, label="Neb")
    plt.plot(
        waves,
        Obsspec,
        label=label_name,
    )
    plt.legend()
    if display:
        plt.show()


def plot_extremes(
    axes,
    waves,
    Ashift,
    Bshift,
    Nebshift,
    Obsspec,
    specsum,
    specname,
    phi,
    MJD,
    pltExtyMin,
    pltExtyMax,
    StarName,
    range_str,
    K1now,
    K2now,
    NebLines=False,
    Panel=0,
    linewid_ext=3,
    extremes_fig_size=(7, 8),
    display=False,
):
    print(linewid_ext)
    axes[Panel].plot(
        waves,
        Ashift,
        label="Prim Dis.",
        color="red",
        linestyle="dotted",
        linewidth=linewid_ext,
    )
    axes[Panel].plot(
        waves, Bshift, label="Sec. Dis.", color="green", linewidth=linewid_ext
    )
    if NebLines:
        axes[Panel].plot(waves, Nebshift, label="Nebular Dis.", color="purple")
    axes[Panel].plot(
        waves,
        Obsspec,
        color="blue",
        label=str(round(MJD, 0)) + r", $\varphi=$" + str(round(phi, 2)),
    )
    axes[Panel].plot(
        waves,
        specsum,
        label="Sum Dis.",
        color="black",
        linestyle="--",
        linewidth=linewid_ext,
    )
    axes[Panel].set_title(specname.split("/")[-1])
    axes[Panel].legend(prop={"size": legsize}, loc=locleg, framealpha=alphaleg)
    axes[Panel].set_ylabel("Normalised flux")
    axes[Panel].set_ylim(pltExtyMin, pltExtyMax)
    DiffMajor = int((waves[-1] - waves[0]) / 3)
    axes[Panel].xaxis.set_major_locator(MultipleLocator(DiffMajor))
    axes[Panel].xaxis.set_minor_locator(MultipleLocator(DiffMajor / 5.0))
    axes[Panel].yaxis.set_minor_locator(MultipleLocator(0.01))

    if Panel == 1:
        plt.tight_layout()
        file_name = f"Output/{StarName}_{range_str}_Extremes_"
        file_name += f"{np.round(K1now)}_{np.round(K2now)}.pdf"
        plt.savefig(file_name, bbox_inches="tight")

        if display:
            plt.show()


def plot_conv(itr, eps, display=False):
    plt.scatter(itr, eps, color="blue")
    plt.ylabel("log(Eps)")
    plt.xlabel("iteration number")
    if display:
        plt.show()


def plot_itr(itr, waves, A, B, Nebspec, display=False):
    plt.plot(waves, A, label=itr)
    plt.plot(waves, B, label=itr)
    if (Nebspec > 0).any():
        plt.plot(waves, Nebspec, label=itr)
    plt.ylabel("Normalised flux")
    plt.xlabel("Wavelength")
    plt.legend()
    if display:
        plt.show()
