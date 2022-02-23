#!/usr/bin/env python

# See Fig3-4.ipynb for details.
# Whyjay Zheng
# File created  Oct 21, 2021
# Last modified Feb 22, 2022

import pejzero
import rasterio
from netCDF4 import Dataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

glacier_file = sys.argv[1]
speed_file = '../data/GRE_G0240_1998_v.tif'
vdiff_file = '../data/GRE_G0240_diff-2018-1998_v.tif'

ds = Dataset(glacier_file, 'r')
flowline_groups, _ = pejzero.get_flowline_groups(ds)
primary_flowlines = [i for i in flowline_groups if 'iter' not in i.path]

results = {}

with rasterio.open(speed_file) as speed_data, rasterio.open(vdiff_file) as vdiff_data:
    for flowline_group in primary_flowlines:

        data_group = pejzero.cal_pej0_for_each_flowline(flowline_group, speed_data, vdiff_data)

        if data_group is not None:
            results[flowline_group.name] = data_group
        
results['avg'] = pejzero.cal_avg_for_each_basin(results)

#### plot results
        
pej0_plot_length = 200
matplotlib.rc('font', size=24)
matplotlib.rc('axes', linewidth=2)
fig, ax3 = plt.subplots(5, 2, sharex=True, figsize=(26, 20))
gs = ax3[1, 1].get_gridspec()
for ax in ax3[:, 1]:
    ax.remove()
axbig = fig.add_subplot(gs[1:4, 1])

for key in results:

    if key != 'avg':
        ax3[0, 0].plot(results[key]['d'], results[key]['s'], color='xkcd:aquamarine', linewidth=2)
        ax3[0, 0].plot(results[key]['d'], results[key]['b'], color='xkcd:brown', linewidth=2)
        ax3[1, 0].plot(results[key]['d'], results[key]['u'], color='xkcd:light green', linewidth=2)
        ax3[2, 0].plot(results[key]['d'], results[key]['pe_ignore_dslope'], color='xkcd:light red', linewidth=2)
        ax3[3, 0].plot(results[key]['d'], results[key]['j0_ignore_dslope'], color='xkcd:light blue', linewidth=2)
        ax3[4, 0].plot(results[key]['d'], results[key]['udiff_sm'], color='xkcd:light grey', linewidth=2)
        axbig.plot(results[key]['pe_ignore_dslope'][:pej0_plot_length], results[key]['j0_ignore_dslope'][:pej0_plot_length], '.-', 
                   color='xkcd:light purple', linewidth=2)
        # plot first non-NaN value (the one closest to the terminus)
        axbig.plot(next(x for x in results[key]['pe_ignore_dslope'][:pej0_plot_length] if not np.isnan(x)),
                   next(x for x in results[key]['j0_ignore_dslope'][:pej0_plot_length] if not np.isnan(x)), '.', color='xkcd:light purple', markersize=25)
    else:
        ax3[1, 0].plot(results[key]['d'], results[key]['u'], color='xkcd:dark green', linewidth=4)
        ax3[2, 0].plot(results[key]['d'], results[key]['pe_ignore_dslope'], color='xkcd:dark red', linewidth=4)
        ax3[3, 0].plot(results[key]['d'], results[key]['j0_ignore_dslope'], color='xkcd:dark blue', linewidth=4)
        ax3[4, 0].plot(results[key]['d'], results[key]['udiff_sm'], color='xkcd:dark grey', linewidth=4)
        axbig.plot(results[key]['pe_ignore_dslope'][:pej0_plot_length], results[key]['j0_ignore_dslope'][:pej0_plot_length], '.-', 
                   color='xkcd:dark purple', linewidth=4, markersize=10)
        # plot first non-NaN value (the one closest to the terminus)
        axbig.plot(next(x for x in results[key]['pe_ignore_dslope'][:pej0_plot_length] if not np.isnan(x)),
                   next(x for x in results[key]['j0_ignore_dslope'][:pej0_plot_length] if not np.isnan(x)), '.', color='xkcd:dark purple', markersize=30)
        
letter_specs = {'fontsize': 30, 'fontweight': 'bold', 'va': 'top', 'ha': 'center'}
ax3[0, 0].set_title(Path(glacier_file).stem)
ax3[0, 0].set_ylabel('Elevantion (m): \n Surface (cyan) \n bed (brown)')
ax3[0, 0].text(0.04, 0.95, 'A', transform=ax3[0, 0].transAxes, **letter_specs)
ax3[1, 0].set_ylabel('Speed 1998 (m yr$^{-1}$)')
ax3[1, 0].text(0.96, 0.95, 'B', transform=ax3[1, 0].transAxes, **letter_specs)
ax3[2, 0].set_ylabel(r'$\frac{P_e}{\ell}$ (m$^{-1}$)')
ax3[2, 0].text(0.96, 0.95, 'C', transform=ax3[2, 0].transAxes, **letter_specs)
ax3[3, 0].set_ylabel(r'$J_0$ (m yr$^{-1}$)')
ax3[3, 0].text(0.96, 0.95, 'D', transform=ax3[3, 0].transAxes, **letter_specs)
ax3[4, 0].set_xlabel('Distance from terminus (km)')
ax3[4, 0].set_ylabel('Speed change \n 1998–2018 (m yr$^{-1}$)')
ax3[4, 0].text(0.96, 0.95, 'E', transform=ax3[4, 0].transAxes, **letter_specs)
axbig.set_xlabel(r'$\frac{P_e}{\ell}$ (m$^{-1}$)')
axbig.set_ylabel(r'$J_0$ (m yr$^{-1}$)')
axbig.set_title('Dot spacing: 50 m; \n Big dot indicates the first non-NaN value \n (closest to the terminus)')
axbig.text(0.03, 0.985, 'F', transform=axbig.transAxes, **letter_specs)
pe_labels = ['{:.6f}'.format(x) for x in axbig.get_xticks()]
axbig.set_xticklabels(pe_labels, rotation=45)

outdir = '../data/results/single_basins/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

plt.savefig(outdir + Path(glacier_file).stem + '.png')