#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy

import numpy as np
from scipy import interpolate


def clean_cosmic_ray(
    spec,
    thold=6,
    cosize=10,
    max_iter=10,
    forbidden_ranges=[
        [3970, 3975],
        [4020, 4030],
        [4103.0, 4108.0],
        [4342, 4347],
        [4365, 4369],
        [4391, 4393],
        [4470, 4477.0],
    ],
):
    waves = copy.deepcopy(spec[:, 0])
    fluxes = copy.deepcopy(spec[:, 1])

    spec_clean = copy.deepcopy(spec)

    for itr in range(max_iter):
        print(f"Cosmic ray cleaning, iteration {itr} of {max_iter}.")

        # masking out regions NOT to be cleaned
        for wrange in forbidden_ranges:
            if "wave_cond_mask" not in locals():
                wave_cond_mask = (waves > wrange[0]) * (waves < wrange[1])
            else:
                wave_cond_mask += (waves > wrange[0]) * (waves < wrange[1])

        fluxes[wave_cond_mask] = 1.0
        fluxdiff = np.append(0, np.diff(fluxes))
        sigma = thold * np.mean(np.absolute(fluxdiff))

        # Find points whose gradients are larger than thold*average
        gradient_condition = np.absolute(fluxdiff) > sigma

        # Weak point are of no interest
        flux_condition = fluxes > 1.0
        posgrad = fluxdiff > 0
        flagged_args = np.where(gradient_condition & flux_condition)[0]

        # Break out if no more cosmic ray is detected
        if np.size(flagged_args) - np.count_nonzero(flagged_args):
            print(
                "No cosmics detected with the given threshhold and size. "
                f"Terminating after {itr} of {max_iter} iterations."
            )

        blimit = 0
        for i in flagged_args:
            if (
                (waves[i] < blimit)
                or (i < cosize)
                or (i > waves.size - cosize)
            ):
                continue

            cosmic = fluxes[i - cosize : i + cosize + 1]

            if posgrad[i]:
                if np.any(cosmic[:cosize] - fluxes[i] > 0):
                    continue
            else:
                if np.any(cosmic[cosize + 1 :] - fluxes[i] > 0):
                    continue

            ipeak = i - cosize + np.argmax(fluxes[i - cosize : i + cosize + 1])
            fpeak = fluxes[ipeak]
            cosmic = fluxes[ipeak - cosize : ipeak + cosize + 1]
            cosb = cosmic[: cosize + 1]
            cosr = cosmic[cosize:]
            fmeadb = np.mean(cosb)
            fmeadr = np.mean(cosr)
            cosbdiff = np.append(np.diff(cosb), 0)
            cosrdiff = np.append(0, np.diff(cosr))
            sigmab = np.mean(np.absolute(cosbdiff))
            sigmar = np.mean(np.absolute(cosrdiff))
            condsmallb = cosb - fmeadb < 0.1 * (fpeak - fmeadb)
            condsmallr = cosr - fmeadr < 0.1 * (fpeak - fmeadr)
            argb = np.where(
                (np.roll(cosbdiff, -1) > sigmab) & (condsmallb) & (cosb > 0.5)
            )[0]
            argr = np.where(
                (cosrdiff < -sigmar) & (condsmallr) & (cosr > 0.5)
            )[0]

            # If nothing needs to be cleaned
            if len(argb) == 0 or len(argr) == 0:
                continue

            argb = ipeak - cosize + argb[-1]
            argr = ipeak + argr[0]

            # If nothing needs to be cleaned
            if (abs(fluxes[argb] - fpeak) < sigmab) or (
                abs(fluxes[argr] - fpeak) < sigmar
            ):
                continue

            spec_clean[argb : argr + 1, 1] = np.interp(
                waves[argb : argr + 1],
                [waves[argb], waves[argr]],
                [fluxes[argb], fluxes[argr]],
            )
            blimit = waves[argr]

    return spec_clean


def normalise(spec, points=[]):
    waves = copy.deepcopy(spec[:, 0])
    fluxes = copy.deepcopy(spec[:, 1])

    cont_indices = np.array(
        [np.argmin(np.abs(waves - points[i])) for i in np.arange(len(points))]
    )
    cont_fluxes = np.array(
        [
            np.average(
                fluxes[
                    max(cont_indices[i] - 7, 0) : min(
                        cont_indices[i] + 7, len(waves)
                    )
                ]
            )
            for i in np.arange(len(cont_indices))
        ]
    )

    cont_spline = interpolate.interp1d(
        waves[cont_indices],
        cont_fluxes,
        bounds_error=False,
        fill_value="extrapolate",
    )(waves)

    spec_norm = np.array([waves, fluxes / cont_spline]).T

    return spec_norm
