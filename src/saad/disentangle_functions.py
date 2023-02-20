#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# shift-and-add & grid disentangling, by Tomer Shenar, with contributions
# from Matthias Fabry & Julia Bodensteiner 21.11.2022, V1.0; feel free to
# contact at T.Shenar@uva.nl or tomer.shenar@gmail.com for questions/inquires
# Algorithm and examples in Gonzales & Levato 2006, A&A, 448, 283;
# Shenar et al. 2021, A&A, 639, 6; Shenar et al. 2022, A&A, 665, 148


import copy

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import gammainc

from .plotting import plot_conv, plot_extremes, plot_fits, plot_itr

__all__ = ["Disentangle"]


class Disentangle:
    def __init__(self, spec_list, orbital_params):
        # Constants
        self.clight = 2.9979e5

        # reference to the input
        self.spec_list = spec_list

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
        self.A = None

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
        P1=0.68,
        comp="secondary",
        parb_size=3,
        display=False,
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

        # plot
        plt.scatter(Ks_comp, redchi2)
        plt.scatter(Ks_comp, chi2 / nu)
        plt.plot(chi2fine, parb, color="orange")
        plt.plot(
            [Ks_comp[0], Ks_comp[-1]],
            [chi2P1, chi2P1],
            color="red",
            label=r"1$\sigma$ contour",
        )
        plt.ylabel(r"Normalised reduced $\chi^2$")
        plt.xlabel(f"${{{K_label}}}$ [km/s]")
        np.savetxt(
            f"Output/{range_str}_grid_dis_{K_name}.txt",
            np.c_[Ks_comp, redchi2],
            header=f"#1sigma = {chi2P1 * parb_min}",
        )
        plt.legend()

        plt.savefig(
            f"Output/{range_str}_Grid_disentangling_{K_name}.pdf",
            bbox_inches="tight",
        )

        if display:
            plt.show()

        return chi2P1, K2min, K2err

    # Ensures that arr values where arr > lim are set to 0 in domains
    # specified in pos_lim array
    def _limit(self, waves, arr, lim, pos_lim):
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
        star_name,
        scaling_neb,
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
        extremes_fig_size=(7, 8),
        display=False,
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
                plot_min_yarr.append(
                    np.amin(
                        interp1d(
                            obs_specs[ind][:, 0],
                            obs_specs[ind][:, 1] - 1,
                            bounds_error=False,
                            fill_value=0.0,
                        )(waves[wave_calc_cond])
                    )
                )
                plot_max_yarr.append(
                    np.amax(
                        interp1d(
                            obs_specs[ind][:, 0],
                            obs_specs[ind][:, 1] - 1,
                            bounds_error=False,
                            fill_value=0.0,
                        )(waves[wave_calc_cond])
                    )
                )
            plot_min_yarr.append(min(A))
            plot_min_yarr.append(min(B))
            plot_max_yarr.append(max(A))
            plot_max_yarr.append(max(B))
            plt_ext_ymin = min(plot_min_yarr) * 1.1
            plt_ext_ymax = max(plot_max_yarr) * 1.1

        for ind in np.arange(len(obs_specs)):
            vA = vrA[ind] / self.clight
            vB = vrB[ind] / self.clight
            fac_shift_1 = np.sqrt((1 + vA) / (1 - vA))
            fac_shift_2 = np.sqrt((1 + vB) / (1 - vB))
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
                # vNeb = 0.0
                neb_shift = (
                    neb_fac
                    * scaling_neb[ind]
                    * interp1d(
                        waves, neb_spec, bounds_error=False, fill_value=0.0
                    )(waves[wave_calc_cond])
                )
                specsum = A_shift + B_shift + neb_shift
            else:
                neb_shift = np.zeros(len(A_shift))
                specsum = A_shift + B_shift
            sigma = (
                np.std(obs_spec[:S2NpixelRange])
                + np.std(obs_spec[-S2NpixelRange:])
            ) / 2.0
            if resid:
                residuals.append(obs_spec - specsum)
            Sum += np.sum((obs_spec - specsum) ** 2 / sigma**2)
            if PLOTFITS:
                # User needs to change kcount==1 condition if a specific K2
                # is desired for plotting.
                if self.kcount == kcount_usr:
                    label_name = (
                        spec_names[ind].split("/")[-1]
                        + r", $\varphi=$"
                        + str(round(phis[ind], 2))
                    )
                    plot_fits(
                        waves=waves[wave_calc_cond],
                        specsum=specsum,
                        A_shift=A_shift,
                        B_shift=B_shift,
                        neb_shift=neb_shift,
                        obs_spec=obs_spec,
                        label_name=label_name,
                        display=display,
                    )

            if PLOTEXTREMES:
                _, axes = plt.subplots(
                    nrows=2, ncols=1, figsize=extremes_fig_size
                )
                if self.kcount == kcount_extremeplot:
                    if ind == min(RV_ext_min_ind, RV_ext_max_ind):
                        _panel = 0
                    elif ind == max(RV_ext_min_ind, RV_ext_max_ind):
                        _panel = 1
                    else:
                        _panel = None

                    # Only plot if ind is 0 or maximum
                    if self.K1now is None:
                        _k1 = self.orbital_params["K1"]
                    else:
                        _k1 = self.K1now

                    if self.K2now is None:
                        _k2 = K2s[self.kcount]
                    else:
                        _k2 = self.K2now

                    if _panel is not None:
                        plot_extremes(
                            axes,
                            waves[wave_calc_cond],
                            A_shift,
                            B_shift,
                            neb_shift,
                            obs_spec,
                            specsum,
                            spec_names[ind],
                            phis[ind],
                            mjds[ind],
                            plt_ext_ymin,
                            plt_ext_ymax,
                            star_name,
                            range_str,
                            _k1,
                            _k2,
                            neb_lines=neb_lines,
                            Panel=_panel,
                            line_wid_ext=line_wid_ext,
                            extremes_fig_size=extremes_fig_size,
                            display=display,
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
        star_name,
        scaling_neb,
        neb_spec,
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
                print("Disentangeling..... K1, K2=", self.K1now, self.K2now)
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
        Ss1 = [
            interp1d(
                obs_specs[i][:, 0] / fac_shift_1[i],
                obs_specs[i][:, 1] - 1.0,
                bounds_error=False,
                fill_value=0.0,
                kind=inter_kind,
            )(waves)
            for i in np.arange(len(obs_specs))
        ]
        Ss2 = [
            interp1d(
                obs_specs[i][:, 0] / fac_shift_2[i],
                obs_specs[i][:, 1] - 1.0,
                bounds_error=False,
                fill_value=0.0,
                kind=inter_kind,
            )(waves)
            for i in np.arange(len(obs_specs))
        ]
        # Frame of Refernce star 1:
        WavesBA = np.array(
            [
                waves * fac_shift_2[i] / fac_shift_1[i]
                for i in np.arange(len(vrads1))
            ]
        )
        # Frame of Refernce star 2:
        WavesAB = np.array(
            [
                waves * fac_shift_1[i] / fac_shift_2[i]
                for i in np.arange(len(vrads1))
            ]
        )
        if neb_lines:
            FacshiftNeb = np.ones(len(vrads1))
            SsNeb = [
                interp1d(
                    obs_specs[i][:, 0] / FacshiftNeb[i],
                    obs_specs[i][:, 1] - 1.0,
                    bounds_error=False,
                    fill_value=0.0,
                    kind=inter_kind,
                )(waves)
                for i in np.arange(len(obs_specs))
            ]
            WavesNebA = np.array(
                [
                    waves * FacshiftNeb[i] / fac_shift_1[i]
                    for i in np.arange(len(vrads1))
                ]
            )
            WavesNebB = np.array(
                [
                    waves * FacshiftNeb[i] / fac_shift_2[i]
                    for i in np.arange(len(vrads1))
                ]
            )
            WavesANeb = np.array(
                [
                    waves * fac_shift_1[i] / FacshiftNeb[i]
                    for i in np.arange(len(vrads1))
                ]
            )
            WavesBNeb = np.array(
                [
                    waves * fac_shift_2[i] / FacshiftNeb[i]
                    for i in np.arange(len(vrads1))
                ]
            )
        itr = 0
        while itr < itr_num_lim:
            itr += 1
            BA_shifts = [
                interp1d(
                    WavesBA[i],
                    B,
                    bounds_error=False,
                    fill_value=0.0,
                    kind=inter_kind,
                )
                for i in np.arange(len(obs_specs))
            ]
            if neb_lines:
                NebA_shifts = [
                    interp1d(
                        WavesNebA[i],
                        neb_spec,
                        bounds_error=False,
                        fill_value=0.0,
                        kind=inter_kind,
                    )
                    for i in np.arange(len(obs_specs))
                ]
                specMean = np.sum(
                    np.array(
                        [
                            weights[i]
                            * (
                                Ss1[i]
                                - BA_shifts[i](waves)
                                - neb_fac
                                * scaling_neb[i]
                                * NebA_shifts[i](waves)
                            )
                            for i in np.arange(len(Ss1))
                        ]
                    ),
                    axis=0,
                )
            else:
                specMean = np.sum(
                    np.array(
                        [
                            weights[i] * (Ss1[i] - BA_shifts[i](waves))
                            for i in np.arange(len(Ss1))
                        ]
                    ),
                    axis=0,
                )
            Anew = interp1d(
                waves,
                specMean,
                bounds_error=False,
                fill_value=0.0,
                kind=inter_kind,
            )(waves)
            if strict_neg_A:
                Anew = self._limit(waves, Anew, pos_lim_all, pos_lim_cond_A)
            if self.A is not None:
                Epsnew = np.amax((self.A - Anew) ** 2)
            else:
                Epsnew = 0.0
            self.A = copy.deepcopy(Anew)
            AB_shifts = [
                interp1d(
                    WavesAB[i],
                    self.A,
                    bounds_error=False,
                    fill_value=0.0,
                    kind=inter_kind,
                )
                for i in np.arange(len(obs_specs))
            ]
            if neb_lines:
                NebB_shifts = [
                    interp1d(
                        WavesNebB[i],
                        neb_spec,
                        bounds_error=False,
                        fill_value=0.0,
                        kind=inter_kind,
                    )
                    for i in np.arange(len(obs_specs))
                ]
                specMean = np.sum(
                    np.array(
                        [
                            weights[i]
                            * (
                                Ss2[i]
                                - AB_shifts[i](waves)
                                - neb_fac
                                * scaling_neb[i]
                                * NebB_shifts[i](waves)
                            )
                            for i in np.arange(len(Ss1))
                        ]
                    ),
                    axis=0,
                )
            else:
                specMean = np.sum(
                    np.array(
                        [
                            weights[i] * (Ss2[i] - AB_shifts[i](waves))
                            for i in np.arange(len(Ss1))
                        ]
                    ),
                    axis=0,
                )
            Bnew = interp1d(
                waves,
                specMean,
                bounds_error=False,
                fill_value=0.0,
                kind=inter_kind,
            )(waves)
            if strict_neg_B:
                Bnew = self._limit(waves, Bnew, pos_lim_all, pos_lim_cond_B)
            Epsnew = max(Epsnew, np.sum((B - Bnew) ** 2))
            B = Bnew
            if neb_lines:
                Aneb_shifts = [
                    interp1d(
                        WavesANeb[i],
                        self.A,
                        bounds_error=False,
                        fill_value=0.0,
                        kind=inter_kind,
                    )
                    for i in np.arange(len(obs_specs))
                ]
                Bneb_shifts = [
                    interp1d(
                        WavesBNeb[i],
                        B,
                        bounds_error=False,
                        fill_value=0.0,
                        kind=inter_kind,
                    )
                    for i in np.arange(len(obs_specs))
                ]
                specMean = np.sum(
                    np.array(
                        [
                            weights[i]
                            * self._limit(
                                waves,
                                SsNeb[i]
                                - Aneb_shifts[i](waves)
                                - Bneb_shifts[i](waves),
                                pos_lim_all,
                                pos_lim_cond_Neb,
                            )
                            for i in np.arange(len(Ss1))
                        ]
                    ),
                    axis=0,
                )
                neb_specnew = interp1d(
                    waves,
                    specMean,
                    bounds_error=False,
                    fill_value=0.0,
                    kind=inter_kind,
                )(waves)
                neb_specnew[neb_specnew < pos_lim_all[0]] = 0.0
                Epsnew = max(Epsnew, np.sum(np.abs(neb_spec - neb_specnew)))
                neb_spec = neb_specnew
            if show_itr:
                if itr % 100 == 0:
                    print(
                        "Finished "
                        + str(itr)
                        + " out of "
                        + str(itr_num_lim)
                        + " iterations ("
                        + str(np.round(itr / itr_num_lim * 100.0, 3))
                        + "%)"
                    )
                    # print("Convergence Epsilon:", Epsnew)
            if PLOTCONV:
                plot_conv(itr, np.log10(Epsnew), display)

            if PLOTITR:
                if itr % n_iteration_plot == 0:
                    plot_itr(itr, waves, self.A, B, neb_spec, display)

        print("Finished after ", itr, " iterations")

        dis_spec_vector = np.array([self.A, B, neb_spec])
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
            star_name,
            scaling_neb,
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
        )

        def grid_disentangling2D(
            self,
            waveRanges,
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
            star_name,
            scaling_neb,
            Ini=None,
            show_itr=False,
            inter_kind="linear",
            itr_num_lim=100,
            NebOff=True,
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
        ):
            """
            Assuming spectrum for secondary (Bini) and vrads1, gamma, and K1,
            explore K2s array for best-fitting K2
            Ini = determines initial assumption for 0'th iteration
            show_itr = determines whether
            """

            diffs = np.zeros(len(K1s) * len(K2s)).reshape(len(K1s), len(K2s))
            self.DoFs = 0
            for waves in waveRanges:
                for k1, K1 in enumerate(K1s):
                    for k2, K2 in enumerate(K2s):
                        orbital_params_updated = copy.deepcopy(
                            self.orbital_params
                        )
                        orbital_params_updated.update({"K1": K1})
                        orbital_params_updated.update({"K2": K2})
                        vrads1, vrads2 = self.v1_and_v2(
                            nus_data, orbital_params_updated
                        )
                        neb_spec = waves * 0.0
                        if Ini == "A":
                            print(
                                "Initial guess provided for component " + Ini
                            )
                            # print Bini(waves)
                            v_first_entry = vrads2
                            v_second_entry = vrads1
                        elif Ini == "B":
                            print(
                                "Initial guess provided for component " + Ini
                            )
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
                            orbital_params_updated,
                            K1s,
                            K2s,
                            mjds,
                            phis,
                            spec_names,
                            range_str,
                            star_name,
                            scaling_neb,
                            neb_spec,
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
            print("True velocities: ", k1min, k2min, K1s[k1min], K2s[k2min])

            # Start with uncertainty on K1:
            if step_size_2 > 0:
                chi2P, K2, K2err = self._chi2con(
                    diffs[k1min, :],
                    self.DoFs,
                    K1s,
                    K2s,
                    range_str,
                    comp="secondary",
                    parb_size=parb_size,
                )
            else:
                K2 = K2s[0]
                K2err = 0
            if step_size_1 > 0:
                chi2P, K1, K1err = self._chi2con(
                    diffs[:, k2min],
                    self.DoFs,
                    K1s,
                    K2s,
                    range_str,
                    comp="primary",
                    parb_size=parb_size,
                )
            else:
                K1 = K1s[0]
                K1err = 0
            print("K1, K1 min error:", K1, K1err)
            print("K2, K2 min error:", K2, K2err)
            return K1, K2
