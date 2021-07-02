# general functions needed for running Pe-J0 comparison for the Greenland Ice Sheet (GrIS)

import numpy as np
from scipy.signal import savgol_filter

def savgol_smoothing(u, elev, bed, w=201, delta=50, mode='interp'):
    h = elev - bed
    h[h < 0] = 0
    elev_sm = my_savgol_filter(elev, window_length=w, polyorder=1, deriv=0, delta=delta, mode=mode)
    bed_sm = my_savgol_filter(bed, window_length=w, polyorder=1, deriv=0, delta=delta, mode=mode)
    u_sm = my_savgol_filter(u, window_length=w, polyorder=1, deriv=0, delta=delta, mode=mode)
    h_sm = my_savgol_filter(h, window_length=w, polyorder=1, deriv=0, delta=delta, mode=mode)
    dudx_sm = my_savgol_filter(u, window_length=w, polyorder=1, deriv=1, delta=delta, mode=mode)
    dhdx_sm = my_savgol_filter(h, window_length=w, polyorder=1, deriv=1, delta=delta, mode=mode)
    alpha_sm = -my_savgol_filter(elev, window_length=w, polyorder=1, deriv=1, delta=delta, mode=mode)
    # slope_rad_sm = -np.arctan(alpha_sm)
    # slope_sm = -alpha_sm
    # slope_rad_sm = alpha_sm
    d2hdx2_sm = my_savgol_filter(h, window_length=w, polyorder=2, deriv=2, delta=delta, mode=mode)
    # dalphadx_sm = my_savgol_filter(slope_rad_sm, window_length=w, polyorder=1, deriv=1, delta=delta, mode=mode)
    dalphadx_sm = my_savgol_filter(elev, window_length=w, polyorder=2, deriv=2, delta=delta, mode=mode)
    return elev_sm, bed_sm, u_sm, h_sm, dudx_sm, dhdx_sm, alpha_sm, d2hdx2_sm, dalphadx_sm

def my_savgol_filter(x, window_length, polyorder=1, deriv=0, delta=50, mode='interp'):
    x_sm = savgol_filter(x, window_length=window_length, polyorder=polyorder, deriv=deriv, delta=delta, mode=mode)
    x_sm[:window_length//2] = np.nan
    x_sm[-window_length//2+1:] = np.nan
    for i in range(window_length//2):
        if i > 60:
            reduced_win_length = i * 2 + 1
            # print(i)
            tmp = savgol_filter(x, window_length=reduced_win_length, polyorder=polyorder, deriv=deriv, delta=delta, mode=mode)
            x_sm[i] = tmp[i]
            x_sm[-i-1] = tmp[-i-1]
    return x_sm

def pe_corefun(u, h, dudx, dhdx, slope, dalphadx, m=3):
    term1 = (m + 1) * slope / (m * h)  #  (m+1)alpha / (mH)
    term2 = -dudx / u                      # -U' / U         
    term3 = -dhdx / h                      # -H' / H
    term4 = dalphadx / slope           # alpha' / alpha
    pe = term1 + term2 + term3 + term4

    dd0dx = m * (h * dudx / slope + u * dhdx / slope - u * h * dalphadx / slope ** 2)
    kinevelo = (m + 1) * u - dd0dx  # (m+1)U - m(HU'/alpha + UH'/alpha - UHalpha'/alpha^2)
    diffu_const = m * u * h / slope    # mUH / alpha; D0
    # dd0dx_obsv = np.gradient(diffu_const, 200)

    # only calculating where ice thickness > 50 m. (all Ture)
    # idx = h_sm > 50

    # dfdx = kinevelo[idx] * dhdx_sm[idx] + diffu_const[idx] * dalphadx_sm[idx]  # C * H' + D * alpha'
    # j_over_c0 = -dfdx / ((m + 1) * dudx_sm[idx])
    j0 = kinevelo * dhdx + diffu_const * dalphadx  # C * H' + D * alpha'
    j_over_c0 = -j0 / ((m + 1) * dudx)
    
    return pe, j0, term1, term2, term3, term4