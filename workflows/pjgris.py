# This file contains functions needed for running Pe-J0 comparison,
# especially for the Greenland Ice Sheet (GrIS).

import numpy as np
from scipy.signal import savgol_filter
from scipy import interpolate
import warnings


def get_flowline_groups(ds):
    '''
    Adopts from Felikson et al., re-distributed under the MIT license.
    You can find the original script (utils.py) at https://doi.org/10.5281/zenodo.4284715
    Reference to cite if used:
    Felikson, D., A. Catania, G., Bartholomaus, T. C., Morlighem, M., &Noël, B. P. Y. (2021). 
    Steep Glacier Bed Knickpoints Mitigate Inland Thinning in Greenland. Geophysical Research Letters, 48(2), 1–10. https://doi.org/10.1029/2020GL090112
    
    Processing the netCDF4 dataset (ds) prepared by the same paper. Data available at https://zenodo.org/record/4284759
    '''
    flowline_groups = list()
    iteration_list = list()
    flowlines = [k for k in ds.groups.keys() if 'flowline' in k]
    for flowline in flowlines:
        flowline_groups.append(ds[flowline])
        iteration_list.append('main')
        
    iterations = [k for k in ds.groups.keys() if 'iter' in k]
    for iteration in iterations:
        flowlines = [k for k in ds[iteration].groups.keys() if 'flowline' in k]
        for flowline in flowlines:
            flowline_groups.append(ds[iteration][flowline])
            iteration_list.append(iteration)
            
    return flowline_groups, iteration_list

def savgol_smoothing(u, elev, bed, w=201, delta=50, mode='interp'):
    '''
    Apply Savitzky–Golay filter to glacier speed (u), surface elevation (elev), and surface elevations (bed) 
    along a flowline, and calculate smoothed surface elevation (elev_sm), bed elevation (bed_sm), 
    speed (u_sm), ice thickness (h_sm), speed derivative to distance (dudx_sm), 
    thickness derivate to distance (dhdx_sm), surface slope (alpha_sm),
    second derivative of thickness to distance (d2hdx2), and surface slope derivate to distance (dalphadx_sm).
    
    Arguments:
    - u:    1-D numpy array with a size of N
    - elev: 1-D numpy array with a size of N
    - bed:  1-D numpy array with a size of N
    - w: see window_length argument in savgol_filter. 
    - delta: delta in savgol_filter. 
    - mode: mode in savgol_filter. 
    
    Returns:
    - all returns are a 1-D numpy array with a size of N
    
    Doc for savgol_filter:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
    '''
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
    '''
    A customized savgol_filter used in savgol_smoothing.
    
    We have to customize the savgol_filter to avoid the edge effect. 
    Generally, this function replace the edge points (within window_length//2 data points) with a np.nan
    and uses a reduced window length for 60 points closest to the edges.
    '''
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
    '''
    Calculate Peclet number using a default flow parameter m=3.
    
    Arguments:
    - u, h, dudx, dhdx, slope, dalphadx: variables from savgol_smoothing. Must be of the same size.
    
    Returns:
    - pe: Peclet number (precisely, Pe devided by the characteristic length)
    - J0: see the paper for definition
    - term1, term2, term3, term4: four components of pe (see the comments below and Equation XX in the paper)
    
    '''
    term1 = (m + 1) * slope / (m * h)  #  (m+1)alpha / (mH)
    term2 = -dudx / u                      # -U' / U         
    term3 = -dhdx / h                      # -H' / H
    term4 = dalphadx / slope           # alpha' / alpha
    pe = term1 + term2 + term3 + term4
    pe_ignore_dslope = term1 + term2 + term3

    dd0dx = m * (h * dudx / slope + u * dhdx / slope - u * h * dalphadx / slope ** 2)
    kinevelo = (m + 1) * u - dd0dx  # (m+1)U - m(HU'/alpha + UH'/alpha - UHalpha'/alpha^2)
    diffu_const = m * u * h / slope    # mUH / alpha; D0
    # dd0dx_obsv = np.gradient(diffu_const, 200)

    # only calculating where ice thickness > 50 m. (all Ture)
    # idx = h_sm > 50

    # dfdx = kinevelo[idx] * dhdx_sm[idx] + diffu_const[idx] * dalphadx_sm[idx]  # C * H' + D * alpha'
    # j_over_c0 = -dfdx / ((m + 1) * dudx_sm[idx])
    # term5 = kinevelo * dhdx              # C * H'
    term5 = (m + 1) * u * dhdx         # C * H'
    term6 = diffu_const * dalphadx       # D * alpha'
    j0 = term5 + term6
    # j0_ignore_dslope = ((m + 1) * u - m * (h * dudx / slope + u * dhdx / slope)) * dhdx
    j0_ignore_dslope = term5[:]
    # j_over_c0 = -j0 / ((m + 1) * dudx)
    
    return pe, j0, term1, term2, term3, term4, term5, term6, pe_ignore_dslope, j0_ignore_dslope

def cal_pej0_for_each_flowline(flowline_obj, speed_data, vdiff_data, size_limit=280, minimum_amount_valid_u=20, savgol_winlength=251):
    '''
    Calculate Pe/J0 for each flowline object.
    
    Arguments:
    - flowline_obj: flowline object.
    - speed_data: rasterio dataset (for calculating Pe/J0)
    - vdiff_data: rasterio dataset (for comparison)
    - size_limit: minimum size to start calculation, otherwise return None
    - minimum_amount_valid_u: minimum amount of valid u measurements, otherwise return None
    - savgol_winlength: Savgol filter window length.
    
    Returns:
    - data group: dict object with the following entries: 'd', 's', 'b', 'u', 'pe', 'j0', 'term1', 'term2', 'term3', 'term4',  'udiff', and 'udiff_sm'
    Note: Pe is stored as the form of Pe/l.
    '''
     
    x = flowline_obj['x'][:]
    y = flowline_obj['y'][:]
    d = flowline_obj['d'][:]
    b = flowline_obj['geometry']['bed']['BedMachine']['nominal']['h'][:]
    s = flowline_obj['geometry']['surface']['GIMP']['nominal']['h'][:]
    # pe_felikson = flowline_group['Pe']['GIMP']['nominal'][:]

    if d.size < size_limit:
        return None   # skip really short glacier flowline

    xytuple = [(m, n) for m, n in zip(x, y)]
    sample_gen = speed_data.sample(xytuple)
    u = np.array([float(record) for record in sample_gen])
    u[u < 0] = np.nan

    if sum(~np.isnan(u)) <= minimum_amount_valid_u:
        return None

    valid_u_d = d[~np.isnan(u)]
    valid_u_u = u[~np.isnan(u)]
    f = interpolate.interp1d(valid_u_d, valid_u_u, bounds_error=False, fill_value=np.nan)
    u_holefilled = f(d.data)

    valid_idx = ~np.isnan(u_holefilled)

    x_valid = x[valid_idx]
    y_valid = y[valid_idx]
    d_valid = d[valid_idx]
    s_valid = s[valid_idx]
    b_valid = b[valid_idx]
    u_valid = u_holefilled[valid_idx]

    if s_valid.size < size_limit:
        return None   # skip really short glacier flowline

    # the point closet to the divide = 0 km
    x_valid = np.flip(x_valid)
    y_valid = np.flip(y_valid)
    s_valid = np.flip(s_valid)
    b_valid = np.flip(b_valid)
    u_valid = np.flip(u_valid)

    s_sm, b_sm, u_sm, h_sm, dudx_sm, dhdx_sm, slope_sm, d2hdx2_sm, dalphadx_sm = savgol_smoothing(u_valid, s_valid, b_valid, w=savgol_winlength)

    pe, j0, term1, term2, term3, term4,term5, term6, pe_ignore_dslope, j0_ignore_dslope = pe_corefun(u_sm, h_sm, dudx_sm, dhdx_sm, slope_sm, dalphadx_sm)

    xytuple2 = [(m, n) for m, n in zip(x_valid, y_valid)]
    sample_gen2 = vdiff_data.sample(xytuple2)
    udiff = np.array([float(record) for record in sample_gen2])
    udiff[udiff < -6000] = np.nan
    udiff_sm = my_savgol_filter(udiff, window_length=151, polyorder=1, deriv=0, delta=50, mode='interp')

    if sum(~np.isnan(udiff_sm)) == 0:
        return None   # skip flowline without available speed change data

    d_valid_km = d_valid / 1000
    # pe *= 10000

    # flip again so that d = 0 km indicates front and points upstream
    pe = np.flip(pe)
    j0 = np.flip(j0)
    s_sm = np.flip(s_sm)
    b_sm = np.flip(b_sm)
    u_sm = np.flip(u_sm)
    term1 = np.flip(term1)
    term2 = np.flip(term2)
    term3 = np.flip(term3)
    term4 = np.flip(term4)
    term5 = np.flip(term5)
    term6 = np.flip(term6)
    udiff = np.flip(udiff)
    udiff_sm = np.flip(udiff_sm)
    pe_ignore_dslope = np.flip(pe_ignore_dslope)
    j0_ignore_dslope = np.flip(j0_ignore_dslope)
    
    data_group = {'d': d_valid_km, 's': s_sm, 'b': b_sm, 'u': u_sm, 'pe': pe, 'j0': j0, 
                  'term1': term1, 'term2': term2, 'term3': term3, 'term4': term4, 'term5': term5, 'term6': term6, 
                  'udiff': udiff, 'udiff_sm': udiff_sm, 'pe_ignore_dslope': pe_ignore_dslope, 'j0_ignore_dslope': j0_ignore_dslope,}
    return data_group
    
def cal_avg_for_each_basin(data_group):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        minlength = min([len(data_group[x]['d']) for x in data_group])
        d_avg = next(iter(data_group.values()))['d'][:minlength]
        s_agg = np.vstack([data_group[x]['s'][:minlength] for x in data_group])
        s_avg = np.nanmean(s_agg, axis=0)
        b_agg = np.vstack([data_group[x]['b'][:minlength] for x in data_group])
        b_avg = np.nanmean(b_agg, axis=0)
        u_agg = np.vstack([data_group[x]['u'][:minlength] for x in data_group])
        u_avg = np.nanmean(u_agg, axis=0)
        pe_agg = np.vstack([data_group[x]['pe'][:minlength] for x in data_group])
        pe_avg = np.nanmean(pe_agg, axis=0)
        j0_agg = np.vstack([data_group[x]['j0'][:minlength] for x in data_group])
        j0_avg = np.nanmean(j0_agg, axis=0)
        term1_agg = np.vstack([data_group[x]['term1'][:minlength] for x in data_group])
        term1_avg = np.nanmean(term1_agg, axis=0)
        term2_agg = np.vstack([data_group[x]['term2'][:minlength] for x in data_group])
        term2_avg = np.nanmean(term2_agg, axis=0)
        term3_agg = np.vstack([data_group[x]['term3'][:minlength] for x in data_group])
        term3_avg = np.nanmean(term3_agg, axis=0)
        term4_agg = np.vstack([data_group[x]['term4'][:minlength] for x in data_group])
        term4_avg = np.nanmean(term4_agg, axis=0)
        term5_agg = np.vstack([data_group[x]['term5'][:minlength] for x in data_group])
        term5_avg = np.nanmean(term5_agg, axis=0)
        term6_agg = np.vstack([data_group[x]['term6'][:minlength] for x in data_group])
        term6_avg = np.nanmean(term6_agg, axis=0)
        udiff_agg = np.vstack([data_group[x]['udiff'][:minlength] for x in data_group])
        udiff_avg = np.nanmean(udiff_agg, axis=0)
        udiff_sm_agg = np.vstack([data_group[x]['udiff_sm'][:minlength] for x in data_group])
        udiff_sm_avg = np.nanmean(udiff_sm_agg, axis=0)
        pe_ignore_dslope_agg = np.vstack([data_group[x]['pe_ignore_dslope'][:minlength] for x in data_group])
        pe_ignore_dslope_avg = np.nanmean(pe_ignore_dslope_agg, axis=0)
        j0_ignore_dslope_agg = np.vstack([data_group[x]['j0_ignore_dslope'][:minlength] for x in data_group])
        j0_ignore_dslope_avg = np.nanmean(j0_ignore_dslope_agg, axis=0)
    
    avg = {'d': d_avg, 's': s_avg, 'b': b_avg, 'u': u_avg, 'pe': pe_avg, 'j0': j0_avg, 
           'term1': term1_avg, 'term2': term2_avg, 'term3': term3_avg, 'term4': term4_avg,  'term5': term5_avg, 'term6': term6_avg, 
           'udiff': udiff_avg, 'udiff_sm': udiff_sm_avg, 'pe_ignore_dslope': pe_ignore_dslope_avg, 'j0_ignore_dslope': j0_ignore_dslope_avg}
    return avg



def cal_pej0_for_each_flowline_raw(d, s, b, u, size_limit=280, minimum_amount_valid_u=20, savgol_winlength=251):
    '''
    Calculate Pe/J0 given d, s, b, and u.
    
    Arguments:
    - d: distance along the flowline, from terminus
    - s: surface elevation
    - b: bed elevation
    - u: speed
    - size_limit: minimum size to start calculation, otherwise return None
    - minimum_amount_valid_u: minimum amount of valid u measurements, otherwise return None
    - savgol_winlength: Savgol filter window length.
    
    Returns:
    - data group: dict object with the following entries: 'd', 's', 'b', 'u', 'pe', 'j0', 'term1', 'term2', 'term3', 'term4',  'udiff', and 'udiff_sm'
    Note: Pe is stored as the form of Pe/l.
    '''
     
    if d.size < size_limit:
        return None   # skip really short glacier flowline

    if sum(~np.isnan(u)) <= minimum_amount_valid_u:
        return None
    
    nonnan_idx = np.logical_and(~np.isnan(s), ~np.isnan(b), ~np.isnan(u))

    if np.sum(nonnan_idx) < size_limit:
        return None   # skip really short glacier flowline

    # the point closet to the divide = 0 km
    s = np.flip(s)
    b = np.flip(b)
    u = np.flip(u)

    s_sm, b_sm, u_sm, h_sm, dudx_sm, dhdx_sm, slope_sm, d2hdx2_sm, dalphadx_sm = savgol_smoothing(u, s, b, w=savgol_winlength)
    
    pe, j0, term1, term2, term3, term4, term5, term6, pe_ignore_dslope, j0_ignore_dslope = pe_corefun(u_sm, h_sm, dudx_sm, dhdx_sm, slope_sm, dalphadx_sm)

    d_km = d / 1000

    # flip again so that d = 0 km indicates front and points upstream
    pe = np.flip(pe)
    j0 = np.flip(j0)
    s_sm = np.flip(s_sm)
    b_sm = np.flip(b_sm)
    u_sm = np.flip(u_sm)
    term1 = np.flip(term1)
    term2 = np.flip(term2)
    term3 = np.flip(term3)
    term4 = np.flip(term4)
    term5 = np.flip(term5)
    term6 = np.flip(term6)
    pe_ignore_dslope = np.flip(pe_ignore_dslope)
    j0_ignore_dslope = np.flip(j0_ignore_dslope)
    
    data_group = {'d': d_km, 's': s_sm, 'b': b_sm, 'u': u_sm, 'pe': pe, 'j0': j0, 
                  'term1': term1, 'term2': term2, 'term3': term3, 'term4': term4, 'term5': term5, 'term6': term6, 
                  'pe_ignore_dslope': pe_ignore_dslope, 'j0_ignore_dslope': j0_ignore_dslope,}
    return data_group