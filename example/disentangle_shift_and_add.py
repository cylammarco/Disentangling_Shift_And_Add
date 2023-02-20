# shift-and-add & grid disentangling, by Tomer Shenar, with contributions from Matthias Fabry & Julia Bodensteiner
# 21.11.2022, V1.0; feel free to contact at T.Shenar@uva.nl or tomer.shenar@gmail.com for questions/inquires
# Algorithm and examples in Gonzales & Levato 2006, A&A, 448, 283; Shenar et al. 2021, A&A, 639, 6; Shenar et al. 2022, A&A, 665, 148
# Current version only applicable for binaries.
# Input: input file and observed spectra.
# Output: chi2 map on K1,K2 plane (if requested) and separated spectra.
# See "input file" for more documentation
# Coming up in upcoming versions: Higher-order multiples and nebular contamination.

import glob
import os
import sys

import numpy as np
from astropy.io import ascii, fits
from Input_disentangle import *
from matplotlib import pyplot as plt
from saad import file_handler as fh
from saad.disentangle_functions import Disentangle
from saad.spectral_functions import clean_cosmic_ray, normalise
from scipy import interpolate

##################################################################
####### READING OF DATA --- USER POTENTIALLY NEEDS TO EDIT #######
##################################################################

if obs_format == "TXT":
    PhaseFiles = ascii.read(os.path.join(obs_path, obs_data_filename))
    mjds = PhaseFiles["MJD"]
    spec_names = np.array(
        [os.path.join(obs_path, el) for el in PhaseFiles["obsname"]]
    )
elif obs_format == "FITS":
    spec_names = glob.glob(os.path.join(obs_path, "*.fits"))
    mjds = np.array([])


############################################################
####### Starts code -- don't touch unless necessary! #######
############################################################


# Avoid weird errors:
if comp_num != 2:
    print("ERROR: Can't handle more than two components with current version.")
    sys.exit("Exit in disentangle_shift_and_add.py")

# vector of light ratios l1, l2, l3, l4
lguess_vec = [lguess1] + lguess_vec

if lguess_vec[1] == 0.0:
    print("light ratio of component 2 cannot be zero")
    sys.exit()
if comp_num == 3:
    if lguess_vec[2] == 0.0:
        print("light ratio of component 3 cannot be zero")
        sys.exit()
elif comp_num == 4:
    if lguess_vec[2] == 0.0 or lguess_vec[3] == 0.0:
        print("light ratio of components 3 or 4 cannot be zero")
        sys.exit()


S2Ns = []
obs_specs = []
# Read spectra and potentially dates
for i, filepath in enumerate(spec_names):
    # If fits, read dates as well
    if obs_format == "FITS":
        header = fits.getheader(filepath)
        mjds = np.append(mjds, header[mjd_header_keyword])

    # read_file returns a 2D array of waves vs. flux for given file path. User can edit "read_file" function if needed.
    spec = fh.read_file(filepath)

    # Small script for cleaning spectra of cosmics; use at own risk!
    if clean_cos:
        print("Cleaning Cosmics...")
        spec_clean = clean_cosmic_ray(np.copy(spec))
    else:
        spec_clean = np.copy(spec)

    # Can also re-normalise spectra... better avoid
    if renormalise and grid_dis:
        spec_norm = normalise(np.copy(spec_clean, points=norm_points))
    else:
        spec_norm = np.copy(spec_clean)

    # Store the loaded spectrum into list
    obs_specs.append(spec_norm)

    # Compute S2N of spectrum in prespecified range
    waves = spec_norm[:, 0]
    fluxes = spec_norm[:, 1]
    S2Nrange = (waves > S2Nblue) * (waves < S2Nred)
    S2Ns.append(1.0 / np.std(spec[S2Nrange, 1]))

S2Ns = np.array(S2Ns)


# Initialise an disentangler
disentangler = Disentangle(obs_specs, orbital_params)


# Form array of "force negativity" conditions
strict_neg = [strict_neg_A, strict_neg_B, strict_neg_C, strict_neg_D]
pos_lim_cond = [
    pos_lim_cond_A,
    pos_lim_cond_B,
    pos_lim_cond_C,
    pos_lim_cond_D,
    pos_lim_cond_Neb,
]


# Compute true anomalies of data
phis_data = (mjds - orbital_params["T0"]) / orbital_params["Period"] - (
    (mjds - orbital_params["T0"]) / orbital_params["Period"]
).astype(int)
MsData = 2 * np.pi * phis_data
EsData = Disentangle.kepler(1.0, MsData, orbital_params["ecc"])
eccfac = np.sqrt((1 + orbital_params["ecc"]) / (1 - orbital_params["ecc"]))
nusdata = 2.0 * np.arctan(eccfac * np.tan(0.5 * EsData))


## Determines by how much "negative" spectra can be above 1.
S2Ns_mean = np.mean(S2Ns) * np.sqrt(len(S2Ns))


# pos_lim_all = [force_neg_sigma/(S2Ns_mean * lguess_vec[0]), force_neg_sigma/(S2Ns_mean * lguess_vec[1])]
pos_lim_all = force_neg_sigma / S2Ns_mean

# weighting for final co-added spectrum
weights = S2Ns**2 / np.sum(S2Ns**2)

# Grid on which the spectra are calculated on (taken here as longest and densenst wavelength grid among spectra):
for i, spec in enumerate(obs_specs):
    delta = np.amin(np.abs(np.diff(spec[:, 0])))
    w1 = spec[0, 0]
    w2 = spec[-1, 0]
    if i == 0:
        delta_min = delta
        w1_min = w1
        w2_max = w2
    else:
        delta_min = min(delta_min, delta)
        w1_min = min(w1_min, w1)
        w2_max = max(w2_max, w2)

wave_grid_all = np.arange(w1_min, w2_max, delta_min)


# Initialize wavelength arrays for individual line disentangling

wave_grid_diff_cond_all = np.array(
    [(wave_grid_all > r[0]) * (wave_grid_all < r[1]) for r in ranges]
)
wave_grid = wave_grid_all[np.sum(wave_grid_diff_cond_all, axis=0).astype(bool)]
wave_ranges = [
    wave_grid_all[el.astype(bool)] for el in wave_grid_diff_cond_all
]


##Initialize array of components
comp_arr = [
    interpolate.interp1d(
        wave_grid, np.zeros_like(wave_grid), bounds_error=False, fill_value=0.0
    )
] * comp_num


# Initialize search arrays for K1, K2, K3, K4
if dense_k_arr[0] == 1:
    K1s = np.array([orbital_params["K1"]])
else:
    K1s = np.linspace(
        ini_fac_k_arr[0] * orbital_params["K1"],
        fin_fac_k_arr[0] * orbital_params["K1"],
        dense_k_arr[0],
    )
if dense_k_arr[1] == 1:
    K2s = np.array([orbital_params["K2"]])
else:
    K2s = np.linspace(
        ini_fac_k_arr[1] * orbital_params["K2"],
        fin_fac_k_arr[1] * orbital_params["K2"],
        dense_k_arr[1],
    )
if dense_k_arr[2] == 1:
    K3s = np.array([orbital_params["K3"]])
else:
    K3s = np.linspace(
        ini_fac_k_arr[2] * orbital_params["K3"],
        fin_fac_k_arr[2] * orbital_params["K3"],
        dense_k_arr[2],
    )
if dense_k_arr[3] == 1:
    K4s = np.array([orbital_params["K4"]])
else:
    K4s = np.linspace(
        ini_fac_k_arr[3] * orbital_params["K4"],
        fin_fac_k_arr[3] * orbital_params["K4"],
        dense_k_arr[3],
    )


if not os.path.exists(output_path):
    os.mkdir(output_path)


# Run main disentangling routine
if grid_dis:
    # Compute RVs for comp1, comp2
    # Compute RVs for comp1, comp2
    vrads1, vrads2 = disentangler.v1_and_v2(nusdata, orbital_params)
    scaling_neb = np.ones(len(vrads1))
    kcount_extremeplot = np.argmin(np.abs(K1s - velo_plot_usr_K1_ext)) * len(
        K2s
    ) + np.argmin(np.abs(K2s - velo_plot_usr_K2_ext))
    kcount_usr = np.argmin(np.abs(K1s - velo_plot_usr_K1)) * len(
        K2s
    ) + np.argmin(np.abs(K2s - velo_plot_usr_K2))
    K1, K2 = disentangler.grid_disentangling2D(
        wave_ranges,
        nusdata,
        comp_arr[1],
        K1s,
        K2s,
        obs_specs,
        weights,
        strict_neg,
        pos_lim_cond,
        pos_lim_all,
        mjds,
        phis_data,
        spec_names,
        range_str,
        output_path,
        star_name,
        scaling_neb,
        ini="B",
        show_itr=False,
        inter_kind=inter_kind,
        itr_num_lim=itr_num_lim,
        PLOTCONV=PLOTCONV,
        PLOTITR=PLOTITR,
        PLOTEXTREMES=PLOTEXTREMES,
        PLOTFITS=PLOTFITS,
        kcount_extremeplot=kcount_extremeplot,
        line_wid_ext=line_wid_ext,
        comp_num=comp_num,
        parb_size=parb_size,
        n_iteration_plot=n_iteration_plot,
        neb_lines=neb_lines,
        neb_fac=1,
        kcount_usr=kcount_usr,
        extremes_fig_size=extremes_fig_size,
    )
    orbital_params["K1"] = K1
    orbital_params["K2"] = K2
else:
    PLOTEXTREMES = False
    PLOTFITS = False
    K2 = orbital_params["K2"]
    K1 = orbital_params["K1"]
    scaling_neb = np.ones(len(phis_data))
    # K2=2*K1

print("K2 found:", K2)
if K2 < 0:
    print("setting K2=2*K1")
    orbital_params["K2"] = 2 * orbital_params["K1"]
print("disentangling...., K1, K2:", orbital_params["K1"], orbital_params["K2"])
vrads1, vrads2 = Disentangle.v1_and_v2(nusdata, orbital_params)
itr_num_lim = num_itr_final
neb_spec = np.zeros_like(wave_grid_all)
dis_spec_vector, redchi2 = disentangler.disentangle(
    np.zeros_like(wave_grid_all),
    vrads1,
    vrads2,
    wave_grid_all,
    obs_specs,
    weights,
    strict_neg,
    pos_lim_cond,
    pos_lim_all,
    nusdata,
    K1s,
    K2s,
    mjds,
    phis_data,
    spec_names,
    range_str,
    output_path,
    star_name,
    scaling_neb,
    neb_spec,
    resid=False,
    reduce=True,
    show_itr=True,
    once=True,
    inter_kind=inter_kind,
    itr_num_lim=itr_num_lim,
    PLOTCONV=PLOTCONV,
    PLOTITR=PLOTITR,
    PLOTEXTREMES=PLOTEXTREMES,
    PLOTFITS=PLOTFITS,
    kcount_extremeplot=0,
    line_wid_ext=line_wid_ext,
    comp_num=comp_num,
    n_iteration_plot=n_iteration_plot,
    neb_lines=neb_lines,
    neb_fac=1,
    extremes_fig_size=extremes_fig_size,
    display=False,
)


# These are the final, scaled spectra:
A, B, neb_spec = dis_spec_vector

A = (A - 1) / lguess_vec[0] + 1.0
B = (B - 1) / lguess_vec[1] + 1.0
if comp_num >= 3:
    C = (C - 1) / lguess_vec[2] + 1.0
if comp_num == 4:
    D = (D - 1) / lguess_vec[3] + 1.0


lg_vec_1 = lguess_vec[1]
_k1 = np.round(orbital_params["K1"], 3)
_k2 = np.round(orbital_params["K2"], 3)

filename_ending = f"{lg_vec_1}_{_k1}_{_k2}.txt"

np.savetxt(
    os.path.join(output_path, "ADIS_lguess2_K1K2=" + filename_ending),
    np.c_[wave_grid_all, A],
)
np.savetxt(
    os.path.join(output_path, "BDIS_lguess2_K1K2=" + filename_ending),
    np.c_[wave_grid_all, B],
)
if neb_lines:
    np.savetxt(
        os.path.join(output_path, "NebDIS_lguess2_K1K2=" + filename_ending),
        np.c_[wave_grid_all, neb_spec],
    )
