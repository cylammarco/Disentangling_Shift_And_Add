#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# shift-and-add & grid disentangling, by Tomer Shenar, with contributions
# from Matthias Fabry & Julia Bodensteiner 21.11.2022, V1.0; feel free to
# contact at T.Shenar@uva.nl or tomer.shenar@gmail.com for questions/inquires
# Algorithm and examples in Gonzales & Levato 2006, A&A, 448, 283;
# Shenar et al. 2021, A&A, 639, 6; Shenar et al. 2022, A&A, 665, 148


import copy
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.special import gammainc
from spectresc import spectres

from .plotting import (
    plot_best_fit,
    plot_chi2,
    plot_convergence,
    plot_extremes,
    plot_iteration,
)

__all__ = ["Disentangle"]


class Disentangle:
    def __init__(self, spec_list, orbital_params):
        # Constants
        self.clight = 2.9979e5

        # reference to the input
        self.spec_list = []
        for _spec in spec_list:
            wave = _spec[:, 0]
            flux = _spec[:, 1]
            mask = np.isnan(flux) | ~np.isfinite(flux)
            _spec = np.column_stack((wave[mask], flux[mask]))
            self.spec_list.append(_spec)

        # store the orbital parameters
        self.orbital_params = orbital_params

        # initiallise other fitting parameters
        self.kcount = None
        self.k1 = None
        self.k2 = None
        self.DoFs = None
        self.K1now = None
        self.K2now = None
        self.K1s = None
        self.K2s = None

    # Solves the Kepler equation
    @staticmethod
    def kepler(E, M, ecc):
        E2 = (M - ecc * (E * np.cos(E) - np.sin(E))) / (1.0 - ecc * np.cos(E))
        eps = np.abs(E2 - E)
        if np.all(eps < 1e-10):
            return E2
        else:
            return Disentangle.kepler(E2, M, ecc)

    # Given true anomaly nu,
    @staticmethod
    def v1_and_v2(nu, orbital_params):
        # grab the params needed
        Gamma = orbital_params["Gamma"]
        ecc = orbital_params["ecc"]
        K1 = orbital_params["K1"]
        K2 = orbital_params["K2"]

        # convert from degrees to radians
        omega = np.radians(orbital_params["omega"])
        # OmegaOut = np.radians(orbital_params['omegaOut'])
        # Omega_2 = np.radians(orbital_params['omega_2'])

        _tmp = np.cos(omega + nu) + ecc * np.cos(omega)

        # get the v1 and v2 velocities
        v1 = Gamma + K1 * _tmp
        v2 = Gamma - K2 * _tmp

        return v1, v2

    # Calculate K corresponding to chi2 minimum by fitting parabola to
    # minimum region + confidence interval (default: 1sig=68%)
    # redchi2: array of reduced chi2, i.e chi2/DoF (DoF = Degrees of Freedom)
    # nu = DoF
    def _chi2con(
        self,
        redchi2,
        nu,
        Ks,
        range_str,
        output_path,
        P1=0.68,
        comp="secondary",
        parb_size=3,
        display=False,
        fig_type="png",
    ):
        if comp == "secondary":
            K_name = "K2"
            K_label = "K_2"
        elif comp == "primary":
            K_name = "K1"
            K_label = "K_1"
        else:
            raise ValueError(f"Unknonw comp: {comp}")

        Ks_comp = copy.deepcopy(Ks)

        # First fit parabola to chi2 distribution to find minimum
        # (ind_min-parb_size, ind_min+parb_size, default parb_size=3)
        ind_min = np.argmin(redchi2)
        i1 = max(0, ind_min - parb_size)
        i2 = min(len(Ks_comp) - 1, ind_min + parb_size)
        a, b, c = np.polyfit(Ks_comp[i1:i2], redchi2[i1:i2], 2)

        if a < 0:
            print(
                "Could not fit sensible parabola (a < 0); try changing the "
                "fitting range by increasing/decreasing parb_size argument."
            )
            return 0.0, Ks_comp[ind_min], 0.0

        parb_min = c - b**2 / 4.0 / a

        # Now compute non-reduced, normalised chi2 distribution
        chi2 = redchi2 * nu / parb_min

        # The probability distribution of (non-reduced) chi2 peaks at nu.
        # Compute array around this value
        xarr = np.arange(nu / 10.0, nu * 10, 1)

        # The cumulative probability distribution of chi^2
        # (i.e., Prob(chi2) < x) with nu DoFs is the regularised
        # incomplete Gamma function Gamma(nu/2, x/2)
        # Hence, we look for the index and x value at which Prob(chi2) < P1
        ys1 = gammainc(nu / 2.0, xarr / 2) - P1
        minarg1 = np.argmin(np.abs(ys1))

        # This is the chi2 value corresponding to P1 (typically 1-sigma)
        chi2P1 = xarr[minarg1] / nu

        # Now fit parabola to reduced chi2, after normalisation:
        a, b, c = np.polyfit(Ks_comp[i1:i2], chi2[i1:i2] / nu, 2)
        chi2fine = np.arange(Ks_comp[i1], Ks_comp[i2], 0.01)
        parb = a * chi2fine**2 + b * chi2fine + c
        K2min = -b / 2.0 / a
        K2err = (
            K2min - (-b - np.sqrt(b**2.0 - 4.0 * a * (c - chi2P1))) / 2.0 / a
        )
        np.savetxt(
            os.path.join(output_path, f"{range_str}_grid_dis_{K_name}.txt"),
            np.c_[Ks_comp, redchi2],
            header=f"#1sigma = {chi2P1 * parb_min}",
        )

        fig_filename = f"{range_str}_Grid_disentangling_{K_name}.pdf"

        # plot
        plot_chi2(
            Ks_comp,
            redchi2,
            chi2,
            nu,
            chi2fine,
            parb,
            chi2P1,
            K_label,
            output_path,
            fig_filename,
            display,
            fig_type,
        )

        return chi2P1, K2min, K2err

    # Ensures that arr values where arr > lim are set to 0 in domains
    # specified in pos_lim array
    @staticmethod
    def _limit(waves, arr, lim, pos_lim):
        for i, Range in enumerate(pos_lim):
            if "PosCond" not in locals():
                PosCond = (pos_lim[i][0] < waves) * (waves < pos_lim[i][1])
            else:
                PosCond += (pos_lim[i][0] < waves) * (waves < pos_lim[i][1])
        NegCond = np.logical_not(PosCond)
        arr[(arr > lim) * NegCond] = 0.0
        return arr

    # Shrinks the wavelength domain on which obs-mod is calculated to avoid
    # edge issues
    def _reduce_waves(self, waves, nus_data, K1max, K2max):
        orbital_params_max = copy.deepcopy(self.orbital_params)
        orbital_params_max.update({"K1": K1max})
        orbital_params_max.update({"K2": K2max})
        vrA, vrB = self.v1_and_v2(nus_data, orbital_params_max)
        indices = np.where(np.diff(waves) > 1.0)
        wave_calc_cond = waves < 0
        if len(indices) == 0:
            lam_min = waves[0] * (1.0 + max(max(vrA), max(vrB)) / self.clight)
            lam_max = waves[-1] * (1.0 + min(min(vrA), min(vrB)) / self.clight)
            wave_calc_cond = (waves > lam_min) * (waves < lam_max)
        else:
            indices = np.append(0, indices[0])
            indices = np.append(indices, len(waves) - 1)
            for j in np.arange(len(indices) - 1):
                lam_min = waves[indices[j] + 1] * (
                    1.0 + max(max(vrA), max(vrB)) / self.clight
                )
                lam_max = waves[indices[j + 1]] * (
                    1.0 + min(min(vrA), min(vrB)) / self.clight
                )
                wave_calc_cond = wave_calc_cond + (waves > lam_min) * (
                    waves < lam_max
                )
        ##reduce nebular lines regions:
        # if NebLineCutchi2 == True:
        # for wrange in pos_lim_cond_C:
        # if 'NebCond' in locals():
        # _tmp = 1+Gamma/self.clight
        # NebCond += (waves > wrange[0]*_tmp) * (waves < wrange[1]*_tmp)
        # else:
        # NebCond = (waves > wrange[0]*_tmp) * (waves < wrange[1]*_tmp)
        # wave_calc_cond *= ~NebCond
        return wave_calc_cond

    # Calculate difference
    def _calc_diffs(
        self,
        dis_spec_vector,
        vrA,
        vrB,
        waves,
        obs_specs,
        nus_data,
        K1s,
        K2s,
        mjds,
        phis,
        spec_names,
        range_str,
        output_path,
        star_name,
        scaling_neb,
        interpolate=False,
        resid=False,
        reduce=False,
        show_itr=False,
        PLOTEXTREMES=False,
        PLOTFITS=False,
        kcount_extremeplot=0,
        line_wid_ext=3,
        comp_num=2,
        neb_lines=False,
        neb_fac=1,
        S2NpixelRange=5,
        kcount_usr=0,
        extremes_fig_size=(8, 8),
        display=False,
        fig_type="png",
    ):
        RV_ext_max_ind, RV_ext_min_ind = np.argmax(vrA), np.argmin(vrA)
        A, B, neb_spec = dis_spec_vector
        self.K1s = K1s
        self.K2s = K2s

        if resid:
            residuals = []

        wave_calc_cond = self._reduce_waves(
            waves, nus_data, self.K1s[-1], self.K2s[-1]
        )
        Sum = 0

        if self.kcount is None:
            self.kcount = 0

        if self.DoFs is None:
            self.DoFs = 1

        if PLOTEXTREMES:
            plot_min_yarr = []
            plot_max_yarr = []
            for ind in np.arange(len(obs_specs)):
                if interpolate:
                    _new_spec = interp1d(
                        obs_specs[ind][:, 0],
                        obs_specs[ind][:, 1] - 1,
                        bounds_error=False,
                        fill_value=0.0,
                    )(waves[wave_calc_cond])
                else:
                    _new_spec = spectres(
                        waves[wave_calc_cond],
                        obs_specs[ind][:, 0],
                        obs_specs[ind][:, 1] - 1,
                        fill=0.0,
                        verbose=False,
                    )

                plot_min_yarr.append(np.amin(_new_spec))
                plot_max_yarr.append(np.amax(_new_spec))

            plot_min_yarr.append(np.nanpercentile(A, 1.0))
            plot_min_yarr.append(np.nanpercentile(B, 1.0))
            plot_max_yarr.append(np.nanpercentile(A, 99.0))
            plot_max_yarr.append(np.nanpercentile(B, 99.0))
            plt_ext_ymin = min(plot_min_yarr) * 0.9
            plt_ext_ymax = max(plot_max_yarr) * 1.1

        if PLOTEXTREMES:
            extremes_A_shift = []
            extremes_B_shift = []
            extremes_neb_shift = []
            extremes_obs_spec = []
            extremes_spec_sum = []
            extremes_spec_names = []
            extremes_phis = []
            extremes_mjds = []
            extremes_plt_ext_ymin = []
            extremes_plt_ext_ymax = []
            extremes_K1 = []
            extremes_K2 = []
            extremes_neb_lines = []

        for ind in np.arange(len(obs_specs)):
            vA = vrA[ind] / self.clight
            vB = vrB[ind] / self.clight
            fac_shift_1 = np.sqrt((1 + vA) / (1 - vA))
            fac_shift_2 = np.sqrt((1 + vB) / (1 - vB))
            if interpolate:
                A_shift = interp1d(
                    waves * fac_shift_1, A, bounds_error=False, fill_value=0.0
                )(waves[wave_calc_cond])
                B_shift = interp1d(
                    waves * fac_shift_2, B, bounds_error=False, fill_value=0.0
                )(waves[wave_calc_cond])
                obs_spec = interp1d(
                    obs_specs[ind][:, 0],
                    obs_specs[ind][:, 1] - 1,
                    bounds_error=False,
                    fill_value=0.0,
                )(waves[wave_calc_cond])
                if neb_lines:
                    _neb_spec = interp1d(
                        waves, neb_spec, bounds_error=False, fill_value=0.0
                    )(waves[wave_calc_cond])
            else:
                A_shift = spectres(
                    waves[wave_calc_cond],
                    waves * fac_shift_1,
                    A,
                    fill=0.0,
                    verbose=False,
                )
                B_shift = spectres(
                    waves[wave_calc_cond],
                    waves * fac_shift_2,
                    B,
                    fill=0.0,
                    verbose=False,
                )
                obs_spec = spectres(
                    waves[wave_calc_cond],
                    obs_specs[ind][:, 0],
                    obs_specs[ind][:, 1] - 1,
                    fill=0.0,
                    verbose=False,
                )
                if neb_lines:
                    _neb_spec = spectres(
                        waves[wave_calc_cond],
                        waves,
                        neb_spec,
                        fill=0.0,
                        verbose=False,
                    )

            if neb_lines:
                # vNeb = 0.0
                neb_shift = neb_fac * scaling_neb[ind] * _neb_spec
                spec_sum = A_shift + B_shift + neb_shift
            else:
                neb_shift = np.zeros(len(A_shift))
                spec_sum = A_shift + B_shift
            sigma = (
                np.std(obs_spec[:S2NpixelRange])
                + np.std(obs_spec[-S2NpixelRange:])
            ) / 2.0
            if resid:
                residuals.append(obs_spec - spec_sum)
            Sum += np.sum((obs_spec - spec_sum) ** 2 / sigma**2)

            if PLOTEXTREMES:
                # Only plot if ind is 0 or maximum
                if self.K1now is None:
                    _K1 = self.orbital_params["K1"]
                else:
                    _K1 = self.K1now

                if self.K2now is None:
                    _K2 = K2s[self.kcount]
                else:
                    _K2 = self.K2now

                if ind == min(RV_ext_min_ind, RV_ext_max_ind):
                    extremes_A_shift.append(A_shift)
                    extremes_B_shift.append(B_shift)
                    extremes_neb_shift.append(neb_shift)
                    extremes_obs_spec.append(obs_spec)
                    extremes_spec_sum.append(spec_sum)
                    extremes_spec_names.append(spec_names[ind])
                    extremes_phis.append(phis[ind])
                    extremes_mjds.append(mjds[ind])
                    extremes_plt_ext_ymin.append(plt_ext_ymin)
                    extremes_plt_ext_ymax.append(plt_ext_ymax)
                    extremes_K1.append(_K1)
                    extremes_K2.append(_K2)
                    extremes_neb_lines.append(neb_lines)

                elif ind == max(RV_ext_min_ind, RV_ext_max_ind):
                    extremes_A_shift.append(A_shift)
                    extremes_B_shift.append(B_shift)
                    extremes_neb_shift.append(neb_shift)
                    extremes_obs_spec.append(obs_spec)
                    extremes_spec_sum.append(spec_sum)
                    extremes_spec_names.append(spec_names[ind])
                    extremes_phis.append(phis[ind])
                    extremes_mjds.append(mjds[ind])
                    extremes_plt_ext_ymin.append(plt_ext_ymin)
                    extremes_plt_ext_ymax.append(plt_ext_ymax)
                    extremes_K1.append(_K1)
                    extremes_K2.append(_K2)
                    extremes_neb_lines.append(neb_lines)

                else:
                    pass

        if PLOTEXTREMES:
            plot_extremes(
                waves[wave_calc_cond],
                extremes_A_shift,
                extremes_B_shift,
                extremes_neb_shift,
                extremes_obs_spec,
                extremes_spec_sum,
                extremes_spec_names,
                extremes_phis,
                extremes_mjds,
                extremes_plt_ext_ymin,
                extremes_plt_ext_ymax,
                output_path,
                star_name,
                range_str,
                extremes_K1,
                extremes_K2,
                neb_lines=neb_lines,
                extremes_fig_size=extremes_fig_size,
                line_wid_ext=line_wid_ext,
                display=display,
                fig_type=fig_type,
            )

        if PLOTFITS:
            # User needs to change kcount==1 condition if a specific K2
            # is desired for plotting.
            if self.kcount == kcount_usr:
                label_name = (
                    spec_names[ind].split(os.path.sep)[-1]
                    + r", $\varphi=$"
                    + str(round(phis[ind], 2))
                )
                plot_best_fit(
                    waves=waves[wave_calc_cond],
                    spec_sum=spec_sum,
                    A_shift=A_shift,
                    B_shift=B_shift,
                    neb_shift=neb_shift,
                    obs_spec=obs_spec,
                    label_name=label_name,
                    output_path=output_path,
                    star_name=star_name,
                    display=display,
                    fig_type=fig_type,
                )

        print("kcount:", self.kcount)

        if self.kcount == 0:
            try:
                self.DoFs += len(waves[wave_calc_cond]) * (len(obs_specs) - 2)
            except:
                pass

        self.kcount += 1

        if not show_itr:
            print(
                "chi2:",
                Sum / (len(waves[wave_calc_cond]) * len(obs_specs) - comp_num),
            )

        if resid:
            return residuals

        if reduce:
            return Sum / (len(waves[wave_calc_cond]) * len(obs_specs) - 1)
        else:
            return Sum

    def disentangle(
        self,
        B,
        vrads1,
        vrads2,
        waves,
        obs_specs,
        weights,
        strict_neg,
        pos_lim_cond,
        pos_lim_all,
        nus_data,
        K1s,
        K2s,
        mjds,
        phis,
        spec_names,
        range_str,
        output_path,
        star_name,
        scaling_neb,
        neb_spec,
        interpolate=False,
        resid=False,
        reduce=False,
        show_itr=False,
        once=False,
        inter_kind="linear",
        itr_num_lim=100,
        PLOTCONV=False,
        PLOTITR=False,
        PLOTEXTREMES=False,
        PLOTFITS=False,
        kcount_extremeplot=0,
        line_wid_ext=3,
        comp_num=2,
        n_iteration_plot=50,
        neb_lines=False,
        neb_fac=1,
        kcount_usr=0,
        extremes_fig_size=(7, 8),
        display=False,
        fig_type="png",
    ):
        """
        Documentation:
        "B" = array, initial guess for flux of "secondary"
        vrads1, vrads2 = RVs of primary, secondary
        waves: wavelength grid on which disentanglement should take place
        NOTE: If initial guess for primary is preferred, roles of primary
        should change, i.e., one should call:
        disentangle(Aini, vrads2, vrads1, waves)
        resid --> returns array of residual spectra between obs and dis1+dis2
        reduce --> Returns reduced chi2
        """

        strict_neg_A, strict_neg_B, strict_neg_C, strict_neg_D = strict_neg
        A = None

        # If convergence plot: allow spectra to be positive for sensible
        # convergence plot:
        if PLOTCONV:
            strict_neg_A, strict_neg_B, strict_neg_C, strict_neg_D = (
                False,
                False,
                False,
                False,
            )
        (
            pos_lim_cond_A,
            pos_lim_cond_B,
            pos_lim_cond_C,
            pos_lim_cond_D,
            pos_lim_cond_Neb,
        ) = pos_lim_cond
        if not once:
            try:
                self.K1now = K1s[self.k1]
                self.K2now = K2s[self.k2]
                print(f"Disentangling... K1, K2 = {self.K1now}, {self.K2now}")
            except:
                pass
        else:
            self.k1, self.k2 = 0, 0
        fac_shift_1 = np.sqrt(
            (1 + vrads1 / self.clight) / (1 - vrads1 / self.clight)
        )
        fac_shift_2 = np.sqrt(
            (1 + vrads2 / self.clight) / (1 - vrads2 / self.clight)
        )
        if interpolate:
            if inter_kind == "linear":
                Ss1 = np.array(
                    [
                        np.interp(
                            waves,
                            _spec[:, 0] / fac_shift_1[i],
                            _spec[:, 1] - 1.0,
                            left=0.0,
                            right=0.0,
                        )
                        for i, _spec in enumerate(obs_specs)
                    ]
                )
                Ss2 = np.array(
                    [
                        np.interp(
                            waves,
                            _spec[:, 0] / fac_shift_2[i],
                            _spec[:, 1] - 1.0,
                            left=0.0,
                            right=0.0,
                        )
                        for i, _spec in enumerate(obs_specs)
                    ]
                )
            elif inter_kind == "cubic":
                Ss1 = np.array(
                    [
                        CubicSpline(
                            _spec[:, 0] / fac_shift_1[i],
                            _spec[:, 1] - 1.0,
                        )(waves)
                        for i, _spec in enumerate(obs_specs)
                    ]
                )
                Ss2 = np.array(
                    [
                        CubicSpline(
                            _spec[:, 0] / fac_shift_2[i],
                            _spec[:, 1] - 1.0,
                        )(waves)
                        for i, _spec in enumerate(obs_specs)
                    ]
                )
            else:
                raise ValueError(
                    f"Unknown inter_kind: {inter_kind}. "
                    "Please choose from 'linear' or 'cubic'"
                )
        else:
            Ss1 = np.array(
                [
                    spectres(
                        waves,
                        _spec[:, 0] / fac_shift_1[i],
                        _spec[:, 1] - 1.0,
                        fill=0.0,
                        verbose=False,
                    )
                    for i, _spec in enumerate(obs_specs)
                ]
            )
            Ss2 = np.array(
                [
                    spectres(
                        waves,
                        _spec[:, 0] / fac_shift_2[i],
                        _spec[:, 1] - 1.0,
                        fill=0.0,
                        verbose=False,
                    )
                    for i, _spec in enumerate(obs_specs)
                ]
            )

        # Frame of Refernce star 1:
        waves_BA = np.outer(waves, fac_shift_2 / fac_shift_1).T

        # Frame of Refernce star 2:
        waves_AB = np.outer(waves, fac_shift_1 / fac_shift_2).T

        if neb_lines:
            fac_shift_neb = np.ones(len(vrads1))
            if interpolate:
                if inter_kind == "linear":
                    SsNeb = np.array(
                        [
                            np.interp(
                                waves,
                                obs_specs[i][:, 0] / fac_shift_neb[i],
                                obs_specs[i][:, 1] - 1.0,
                                left=0.0,
                                right=0.0,
                            )
                            for i in np.arange(len(obs_specs))
                        ]
                    )
                elif inter_kind == "cubic":
                    SsNeb = np.array(
                        [
                            CubicSpline(
                                obs_specs[i][:, 0] / fac_shift_neb[i],
                                obs_specs[i][:, 1] - 1.0,
                            )(waves)
                            for i in np.arange(len(obs_specs))
                        ]
                    )
                else:
                    raise ValueError(
                        f"Unknown inter_kind: {inter_kind}. "
                        "Please choose from 'linear' or 'cubic'"
                    )
            else:
                SsNeb = np.array(
                    [
                        spectres(
                            waves,
                            _spec[:, 0] / fac_shift_neb[i],
                            _spec[:, 1] - 1.0,
                            fill=0.0,
                            verbose=False,
                        )
                        for i, _spec in enumerate(obs_specs)
                    ]
                )
            waves_neb_A = np.outer(waves, fac_shift_neb / fac_shift_1).T
            waves_neb_B = np.outer(waves, fac_shift_neb / fac_shift_2).T
            waves_A_neb = np.outer(waves, fac_shift_1 / fac_shift_neb).T
            waves_B_neb = np.outer(waves, fac_shift_2 / fac_shift_neb).T

        itr = 0
        eps_itr = []
        while itr < itr_num_lim:
            itr += 1
            if interpolate:
                if inter_kind == "linear":
                    BA_shifts = np.array(
                        [
                            np.interp(
                                waves, waves_BA[i], B, left=0.0, right=0.0
                            )(waves)
                            for i in np.arange(len(obs_specs))
                        ]
                    )
                elif inter_kind == "cubic":
                    BA_shifts = np.array(
                        [
                            CubicSpline(
                                waves_BA[i],
                                B,
                            )(waves)
                            for i in np.arange(len(obs_specs))
                        ]
                    )
                else:
                    raise ValueError(
                        f"Unknown inter_kind: {inter_kind}. "
                        "Please choose from 'linear' or 'cubic'"
                    )
            else:
                BA_shifts = np.array(
                    [
                        spectres(
                            waves,
                            waves_BA[i],
                            B,
                            fill=0.0,
                            verbose=False,
                        )
                        for i in np.arange(len(obs_specs))
                    ]
                )
            if neb_lines:
                if interpolate:
                    if inter_kind == "linear":
                        neb_A_shifts = np.array(
                            [
                                np.interp(
                                    waves,
                                    waves_neb_A[i],
                                    neb_spec,
                                    left=0.0,
                                    right=0.0,
                                )
                                for i in np.arange(len(obs_specs))
                            ]
                        )
                    elif inter_kind == "cubic":
                        neb_A_shifts = np.array(
                            [
                                CubicSpline(
                                    waves_neb_A[i],
                                    neb_spec,
                                )(waves)
                                for i in np.arange(len(obs_specs))
                            ]
                        )
                    else:
                        raise ValueError(
                            f"Unknown inter_kind: {inter_kind}. "
                            "Please choose from 'linear' or 'cubic'"
                        )
                else:
                    neb_A_shifts = np.array(
                        [
                            spectres(
                                waves,
                                waves_neb_A[i],
                                neb_spec,
                                fill=0.0,
                                verbose=False,
                            )
                            for i in np.arange(len(obs_specs))
                        ]
                    )
                spec_mean_A = np.sum(
                    weights[:, None]
                    * (Ss1 - BA_shifts - neb_fac * scaling_neb * neb_A_shifts),
                    axis=0,
                )
            else:
                spec_mean_A = np.sum(
                    weights[:, None] * (Ss1 - BA_shifts), axis=0
                )

            A_new = spec_mean_A
            A_new[np.isnan(A_new) | ~np.isfinite(A_new)] = 0.0

            if strict_neg_A:
                A_new = self._limit(waves, A_new, pos_lim_all, pos_lim_cond_A)
            if A is not None:
                eps_new = np.amax((A - A_new) ** 2)
            else:
                eps_new = 0.0
            A = copy.deepcopy(A_new)
            if interpolate:
                if inter_kind == "linear":
                    AB_shifts = np.array(
                        [
                            np.interp(
                                waves, waves_AB[i], A, left=0.0, right=0.0
                            )(waves)
                            for i in np.arange(len(obs_specs))
                        ]
                    )
                elif inter_kind == "cubic":
                    AB_shifts = np.array(
                        [
                            CubicSpline(
                                waves_AB[i],
                                A,
                            )(waves)
                            for i in np.arange(len(obs_specs))
                        ]
                    )
                else:
                    raise ValueError(
                        f"Unknown inter_kind: {inter_kind}. "
                        "Please choose from 'linear' or 'cubic'"
                    )
            else:
                AB_shifts = np.array(
                    [
                        spectres(
                            waves,
                            waves_AB[i],
                            A,
                            fill=0.0,
                            verbose=False,
                        )
                        for i in np.arange(len(obs_specs))
                    ]
                )

            if neb_lines:
                if interpolate:
                    if inter_kind == "linear":
                        neb_B_shifts = np.array(
                            [
                                np.interp(
                                    waves,
                                    waves_neb_B[i],
                                    neb_spec,
                                    left=0.0,
                                    right=0.0,
                                )
                                for i in np.arange(len(obs_specs))
                            ]
                        )
                    elif inter_kind == "cubic":
                        neb_B_shifts = np.array(
                            [
                                CubicSpline(
                                    waves_neb_B[i],
                                    neb_spec,
                                )(waves)
                                for i in np.arange(len(obs_specs))
                            ]
                        )
                    else:
                        raise ValueError(
                            f"Unknown inter_kind: {inter_kind}. "
                            "Please choose from 'linear' or 'cubic'"
                        )
                else:
                    neb_B_shifts = np.array(
                        [
                            spectres(
                                waves,
                                waves_neb_B[i],
                                neb_spec,
                                fill=0.0,
                                verbose=False,
                            )
                            for i in np.arange(len(obs_specs))
                        ]
                    )
                spec_mean_B = np.sum(
                    weights[:, None]
                    * (Ss2 - AB_shifts - neb_fac * scaling_neb * neb_B_shifts),
                    axis=0,
                )
            else:
                spec_mean_B = np.sum(
                    weights[:, None] * (Ss2 - AB_shifts), axis=0
                )
            B_new = spec_mean_B
            B_new[np.isnan(B_new) | ~np.isfinite(B_new)] = 0.0

            if strict_neg_B:
                B_new = self._limit(waves, B_new, pos_lim_all, pos_lim_cond_B)
            eps_new = max(eps_new, np.sum((B - B_new) ** 2))
            B = B_new
            if neb_lines:
                if interpolate:
                    if inter_kind == "linear":
                        A_neb_shifts = np.array(
                            [
                                np.interp(
                                    waves,
                                    waves_A_neb[i],
                                    A,
                                    left=0.0,
                                    right=0.0,
                                )
                                for i in np.arange(len(obs_specs))
                            ]
                        )
                        B_neb_shifts = np.array(
                            [
                                np.interp(
                                    waves,
                                    waves_B_neb[i],
                                    B,
                                    left=0.0,
                                    right=0.0,
                                )
                                for i in np.arange(len(obs_specs))
                            ]
                        )
                    elif inter_kind == "cubic":
                        A_neb_shifts = np.array(
                            [
                                CubicSpline(
                                    waves_A_neb[i],
                                    A,
                                )(waves)
                                for i in np.arange(len(obs_specs))
                            ]
                        )
                        B_neb_shifts = np.array(
                            [
                                CubicSpline(
                                    waves_B_neb[i],
                                    B,
                                )(waves)
                                for i in np.arange(len(obs_specs))
                            ]
                        )
                    else:
                        raise ValueError(
                            f"Unknown inter_kind: {inter_kind}. "
                            "Please choose from 'linear' or 'cubic'"
                        )
                else:
                    A_neb_shifts = np.array(
                        [
                            spectres(
                                waves,
                                waves_A_neb[i],
                                A,
                                fill=0.0,
                                verbose=False,
                            )
                            for i in np.arange(len(obs_specs))
                        ]
                    )
                    B_neb_shifts = np.array(
                        [
                            spectres(
                                waves,
                                waves_B_neb[i],
                                B,
                                fill=0.0,
                                verbose=False,
                            )
                            for i in np.arange(len(obs_specs))
                        ]
                    )
                spec_mean = np.sum(
                    np.array(
                        [
                            weights[i]
                            * self._limit(
                                waves,
                                SsNeb[i] - A_neb_shifts[i] - B_neb_shifts[i],
                                pos_lim_all,
                                pos_lim_cond_Neb,
                            )
                            for i in np.arange(len(Ss1))
                        ]
                    ),
                    axis=0,
                )

                neb_spec_new = spec_mean
                neb_spec_new[
                    np.isnan(neb_spec_new) | ~np.isfinite(neb_spec_new)
                ] = 0.0
                neb_spec_new[neb_spec_new < pos_lim_all[0]] = 0.0
                eps_new = max(eps_new, np.sum(np.abs(neb_spec - neb_spec_new)))
                neb_spec = neb_spec_new
            if show_itr:
                if itr % 100 == 0:
                    print(
                        f"Finished {itr} out of {itr_num_lim} iterations ("
                        f"{np.round(itr / itr_num_lim * 100.0, 3)}%)"
                    )
                    # print("Convergence Epsilon:", eps_new)

            if PLOTCONV:
                eps_itr.append(np.log10(eps_new))

            if PLOTITR:
                if itr % n_iteration_plot == 0:
                    plot_iteration(
                        itr,
                        waves,
                        A,
                        B,
                        neb_spec,
                        output_path,
                        star_name,
                        display,
                        fig_type,
                    )

        print(f"Finished after {itr} iterations")

        if PLOTCONV:
            plot_convergence(
                eps_itr,
                output_path,
                star_name,
                display,
                fig_type,
            )

        dis_spec_vector = np.array([A, B, neb_spec])

        return dis_spec_vector + 1.0, self._calc_diffs(
            dis_spec_vector,
            vrads1,
            vrads2,
            waves,
            obs_specs,
            nus_data,
            K1s,
            K2s,
            mjds,
            phis,
            spec_names,
            range_str,
            output_path,
            star_name,
            scaling_neb,
            interpolate=interpolate,
            resid=resid,
            reduce=reduce,
            show_itr=show_itr,
            PLOTEXTREMES=PLOTEXTREMES,
            PLOTFITS=PLOTFITS,
            kcount_extremeplot=kcount_extremeplot,
            line_wid_ext=line_wid_ext,
            comp_num=comp_num,
            neb_lines=neb_lines,
            neb_fac=neb_fac,
            kcount_usr=kcount_usr,
            extremes_fig_size=extremes_fig_size,
            display=display,
            fig_type=fig_type,
        )

    def grid_disentangling2D(
        self,
        wave_ranges,
        nus_data,
        Bini,
        K1s,
        K2s,
        obs_specs,
        weights,
        strict_neg,
        pos_lim_cond,
        pos_lim_all,
        mjds,
        phis,
        spec_names,
        range_str,
        output_path,
        star_name,
        scaling_neb,
        interpolate=False,
        ini=None,
        show_itr=False,
        inter_kind="linear",
        itr_num_lim=100,
        PLOTCONV=False,
        PLOTITR=False,
        PLOTEXTREMES=False,
        PLOTFITS=False,
        kcount_extremeplot=0,
        line_wid_ext=3,
        comp_num=2,
        parb_size=3,
        n_iteration_plot=50,
        neb_lines=False,
        neb_fac=1,
        kcount_usr=0,
        extremes_fig_size=(7, 8),
        display=False,
        fig_type="png",
    ):
        """
        Assuming spectrum for secondary (Bini) and vrads1, gamma, and K1,
        explore K2s array for best-fitting K2
        Ini = determines initial assumption for 0'th iteration
        show_itr = determines whether
        """

        diffs = np.zeros(len(K1s) * len(K2s)).reshape(len(K1s), len(K2s))
        self.DoFs = 0
        self.kcount = 0
        for waves in wave_ranges:
            self.kcount = 0
            for k1, K1 in enumerate(K1s):
                for k2, K2 in enumerate(K2s):
                    self.k1 = k1
                    self.k2 = k2
                    orbital_params_updated = copy.deepcopy(self.orbital_params)
                    orbital_params_updated.update({"K1": K1})
                    orbital_params_updated.update({"K2": K2})
                    vrads1, vrads2 = self.v1_and_v2(
                        nus_data, orbital_params_updated
                    )
                    neb_spec = waves * 0.0
                    if ini == "A":
                        print("Initial guess provided for component " + ini)
                        # print Bini(waves)
                        v_first_entry = vrads2
                        v_second_entry = vrads1
                    elif ini == "B":
                        print("Initial guess provided for component " + ini)
                        v_first_entry = vrads1
                        v_second_entry = vrads2
                    else:
                        print(
                            "No initial approximation given, assuming flat"
                            "spectrum for secondary..."
                        )
                        Bini = interp1d(
                            waves,
                            np.ones(len(waves)),
                            bounds_error=False,
                            fill_value=1.0,
                        )
                        v_first_entry = vrads2
                        v_second_entry = vrads1

                    diffs[k1, k2] += self.disentangle(
                        Bini(waves),
                        v_first_entry,
                        v_second_entry,
                        waves,
                        obs_specs,
                        weights,
                        strict_neg,
                        pos_lim_cond,
                        pos_lim_all,
                        nus_data,
                        K1s,
                        K2s,
                        mjds,
                        phis,
                        spec_names,
                        range_str,
                        output_path,
                        star_name,
                        scaling_neb,
                        neb_spec,
                        interpolate=interpolate,
                        inter_kind=inter_kind,
                        itr_num_lim=itr_num_lim,
                        PLOTCONV=PLOTCONV,
                        PLOTITR=PLOTITR,
                        PLOTEXTREMES=PLOTEXTREMES,
                        PLOTFITS=PLOTFITS,
                        kcount_extremeplot=kcount_extremeplot,
                        line_wid_ext=line_wid_ext,
                        comp_num=comp_num,
                        n_iteration_plot=n_iteration_plot,
                        neb_lines=neb_lines,
                        neb_fac=1,
                        kcount_usr=kcount_usr,
                        extremes_fig_size=extremes_fig_size,
                        display=display,
                        fig_type=fig_type,
                    )[-1]

        diffs /= self.DoFs

        try:
            step_size_1 = K1s[1] - K1s[0]
        except:
            step_size_1 = 0

        try:
            step_size_2 = K2s[1] - K2s[0]
        except:
            step_size_2 = 0

        np.savetxt(
            "Output/" + range_str + "_" + "grid_dis_K1K2.txt",
            np.array(diffs),
            header="#K1min, K2min, stepK1, stepK2, DoF = "
            + f"{K1s[0]}, {K2s[0]}, {step_size_1}, {step_size_2}, "
            + f"{self.DoFs}",
        )
        k1min, k2min = np.argwhere(diffs == np.min(diffs))[0]
        # print diffs
        print(
            "True velocities: ",
            k1min,
            k2min,
            np.round(K1s[k1min], 2),
            np.round(K2s[k2min], 2),
        )

        # Start with uncertainty on K1:
        if step_size_2 > 0:
            chi2P2, K2, K2err = self._chi2con(
                diffs[k1min, :],
                self.DoFs,
                K2s,
                range_str,
                output_path,
                P1=0.68,
                comp="secondary",
                parb_size=parb_size,
                display=display,
            )
        else:
            K2 = K2s[0]
            K2err = 0

        if step_size_1 > 0:
            chi2P1, K1, K1err = self._chi2con(
                diffs[:, k2min],
                self.DoFs,
                K1s,
                range_str,
                output_path,
                P1=0.68,
                comp="primary",
                parb_size=parb_size,
                display=display,
            )
        else:
            K1 = K1s[0]
            K1err = 0

        print("K1, K1 min error:", K1, K1err)
        print("K2, K2 min error:", K2, K2err)

        return K1, K2
