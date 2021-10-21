#!/usr/bin/env python
# From Denis Felikson

import sys

import glob
import os
import pickle

import numpy as np
from statsmodels import robust
import operator
import re

from matplotlib import pyplot as plt
from matplotlib import colors
import cmocean
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

import seaborn as sns
import pandas as pd

import re


def get_flowline_groups(ds): #{{{
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
#}}}

def moving_average(x, y, window): #{{{
   ymvavg  = np.array(np.nan * np.ones(y.shape))
   ystddev = np.array(np.nan * np.ones(y.shape))

   for idx, element in enumerate(y):
      if np.isnan(window):
         continue

      # Find elements within window
      idx_start = int(np.max( (0     , np.floor(idx - window/2)) ))
      idx_stop  = int(np.min( (len(y), np.floor(idx + window/2)) ))
      ywindow   = y[idx_start:idx_stop]

      # Find non-nan entries
      valididx = ~np.isnan(ywindow)

      # Average over window
      if np.any(valididx):
         ymvavg[idx]  = ywindow[valididx].mean()
         ystddev[idx] = ywindow[valididx].std()
      else:
         ymvavg[idx]  = np.nan
         ystddev[idx] = np.nan

      # Insert nans into smoothed array where original array had nans
      ymvavg[np.isnan(y)]  = np.nan
      ystddev[np.isnan(y)] = np.nan

   return ymvavg, ystddev
#}}}

def get_Pe3(ID, x, y, d, Pe1, Pe2, PeThreshold=3.0): #{{{
   lastvalid = np.where(~np.isnan(Pe1))[0]
   if len(lastvalid) == 0:
      lastvalid = 0
   else:
      lastvalid = lastvalid[-1]
   Pe = np.hstack( (Pe1[:lastvalid], Pe2[lastvalid:]) )

   nanidx = np.isnan(Pe)
   Pe[nanidx] = -9999.
   idx = np.where(Pe >= PeThreshold)[0]
   Pe[nanidx] = np.nan
   if len(idx) > 0:
      PeX = x[np.min(idx)]
      PeY = y[np.min(idx)]
      PeD = d[np.min(idx)]
   else:
      parts = ID.split('_')
      if len(parts) == 2:
         centerlineID, acrossID = parts
         PeD = getFlowlineLength(centerlineID, acrossID)
      if len(parts) == 3:
         centerlineID, acrossID, iterID = parts
         PeD = getFlowlineLength(centerlineID, acrossID, iterID=iterID)
      PeX = x[np.argmin( np.abs(d - PeD) )]
      PeY = y[np.argmin( np.abs(d - PeD) )]
   
   return PeX, PeY, PeD
   #}}}

def get_percentUnitVol89(x, y, d, dh, dhcumpercentThreshold=89.): #{{{
   dhavg, dhstddev = moving_average(d, dh, 20)
   dhmax = np.nanmin(dhavg)
   dhpercent = 100.0 * (dhavg/dhmax)

   dhthinning = np.where(dhavg < 0, dhavg, np.nan)
   dhcumtotal = np.nansum(dhthinning)
   dhcumsum = np.cumsum(np.where(~np.isnan(dhthinning), dhthinning, 0))
   dhcumpercent = 100.0 * (dhcumsum / dhcumtotal)

   idx = np.where(dhcumpercent >= dhcumpercentThreshold)[0]
   if len(idx) > 0:
      x_threshold = x[np.min(idx)]
      y_threshold = y[np.min(idx)]
      d_threshold = d[np.min(idx)]
   else:
      x_threshold = np.nan
      y_threshold = np.nan
      d_threshold = np.nan

   return d_threshold
#}}}

def get_stats(PeX, PeY, PeD): #{{{
   statsDict = dict()
   statsDict['maxDkey'] = max(PeD.items(), key=operator.itemgetter(1))[0]
   statsDict['minDkey'] = min(PeD.items(), key=operator.itemgetter(1))[0]
   statsDict['min_d'] = PeD[statsDict['minDkey']] #np.min(PeD.values())
   statsDict['max_d'] = PeD[statsDict['maxDkey']] #np.max(PeD.values())
   #statsDict['max_len'] = lengths[maxDkey] #np.max(PeD.values())
   ID = statsDict['maxDkey'].split('_')
   if len(ID) == 2:
      centerlineID, acrossID = ID
      statsDict['max_len'] = getFlowlineLength(centerlineID, acrossID)
   elif len(ID) == 3:
      centerlineID, acrossID, iterID = ID
      statsDict['max_len'] = getFlowlineLength(centerlineID, acrossID, iterID)
   statsDict['max_x'] = PeX[statsDict['maxDkey']]
   statsDict['max_y'] = PeY[statsDict['maxDkey']]
   statsDict['mean'] = np.mean(list(PeD.values()))
   statsDict['median'] = np.median(list(PeD.values()))
   statsDict['rng'] = statsDict['max_d'] - PeD[statsDict['minDkey']]
   statsDict['q75'], statsDict['q25'] = np.percentile(list(PeD.values()), [75 ,25])
   statsDict['iqr'] = statsDict['q75'] - statsDict['q25'];
   statsDict['nmad'] = robust.mad(list(PeD.values()), axis=0)
   statsDict['std'] = np.std(list(PeD.values()))
   # outliers based on std
   #outlier_keys = [k for (k,v) in PeD.items() if v < statsDict['mean']-3*statsDict['std'] or v > statsDict['mean']+3*statsDict['std']]
   # outliers based on iqr (same way that the box plots find outliers)
   outlier_keys = [k for (k,v) in PeD.items() if v < statsDict['q25']-1.5*statsDict['iqr'] or v > statsDict['q75']+1.5*statsDict['iqr']]
   if len(outlier_keys) == 0:
      statsDict['nout'] = 0
      statsDict['outliers'] = []
   else:
      statsDict['nout'] = len(outlier_keys)
      statsDict['outliers'] = outlier_keys
   statsDict['n'] = len(PeD.values())

   return statsDict
   #}}}

def violin_plot(diffs, basins, basins_unique, plotfile, plottype='num', cmapNorm='log', IQRlines=None, highlight=None, highlightLegend=False, shading=None): #{{{
   diffs_all  = [v1 for k0,v0 in diffs.items() for k1,v1 in v0.items()]
   basins_all = basins.values()
   diffs_min = np.min(diffs_all)
   diffs_max = np.max(diffs_all)
   gridx = np.arange(-0.5, len(basins_unique), 1)
   ystep = 100
   gridy = np.linspace(diffs_min, diffs_max, ystep)
   
   diffs_to_grid = list()
   basins_to_grid = list()
   for k0,v0 in diffs.items():
      for k1,v1 in v0.items():
         diffs_to_grid.append(v1)
         basins_to_grid.append(basins_unique.index(basins[k0]))

   fig, ax = plt.subplots(figsize=(15/2.54, 12/2.54))
   ax = sns.violinplot(x=basins_to_grid, y=diffs_to_grid, ax=ax)
   #ax = sns.boxplot(x=basins_to_grid, y=diffs_to_grid, orient='v', ax=ax)
   plt.setp(ax.collections, alpha=.5)

   if highlight:
      #{{{
      df = pd.DataFrame(columns=['flowline', 'across', 'basin', 'diff', 'hue'])
      for basin_placeholder in np.arange(len(basins_unique)):
         df = df.append({'flowline': 'glacier', 'across': 'across', 'basin': basin_placeholder, 'diff': np.nan, 'hue': ''}, ignore_index=True)
      for glacier in highlight:
         hue = glacier
         for across in diffs[glacier].keys():
            df = df.append({'flowline': glacier, 'across': across, 'basin': basins_unique.index(basins[glacier]), 'diff': diffs[glacier][across], 'hue': hue}, ignore_index=True)

      sns.stripplot(x="basin", y="diff", data=df, hue='hue', jitter=True, size=6, palette=((0, 0.4470, 0.74101), (0.850, 0.3250, 0.0980)), ax=ax)
      #}}}

   if IQRlines:
   #{{{
      q1, q2 = np.percentile(diffs_to_grid,IQRlines)
      ax.plot(ax.get_xlim(), [q1, q1], 'k--')
      ax.plot(ax.get_xlim(), [q2, q2], 'k--')
      less = len(np.where(np.array(diffs_to_grid) < q1)[0])
      more = len(np.where(np.array(diffs_to_grid) > q2)[0])
      sys.stdout.write('   {:3.0f}th %ile: {:+4.1f}\n'.format(IQRlines[0], q1))
      sys.stdout.write('   {:3.0f}th %ile: {:+4.1f}\n'.format(IQRlines[1], q2))
   #}}}

   ax.set_ylim( (-200.,900.) )
   if shading:
   #{{{
      yl = ax.get_ylim()
      for b in shading:
         try:
            x = basins_unique.index(b)
            ax.fill_between( [x-0.5, x+0.5], [yl[0], yl[0]], [yl[1], yl[1]], facecolor='gray', alpha=0.5, zorder=-9999)
         except:
            pass
   #}}}

   ax.tick_params(labelsize=18)
   ax.set_xticks(np.arange(0., len(basins_unique), 1))
   ax.set_xticklabels(basins_unique)
   ax.set_ylabel('distance between predicted\nthinning limit and sea level (km)', {'size':18})
   
   if highlight: ax.get_legend().remove()

   #plt.show()
   #import pdb; pdb.set_trace()
   fig.savefig(plotfile, bbox_inches='tight')
   plt.close(fig)
#}}}

def getFlowlineLength(centerlineID, acrossID, iterID=None): #{{{
   length = None

   if iterID is None or iterID == 'main':
      f = open('clean_flowline_lengths.txt', 'r')
      flowlineID = '_'.join( (centerlineID, acrossID) )
   else:
      f = open('clean_iterated_flowline_lengths.txt', 'r')
      flowlineID = '_'.join( (centerlineID, iterID, acrossID) )

   for line in f:
      if re.search(flowlineID, line):
         length_str = line.split(',')[1]

         length = float(line.split(',')[1])

   f.close()
   return length
#}}}

def glacier_wide_thinning_limit(statsDict, calc=None): #{{{
   if calc == 'mean':
      return statsDict['mean']
   elif calc == 'max':
      return statsDict['max_d']
   else:
      return statsDict['max_d'] - statsDict['std']
#}}}

def glacier_wide_thinning_limits_with_errors(errors_dict, PeThreshold, calc=None, remove_outliers=False): #{{{
   # Error/warning string (append to this then write at the end)
   error_warning_string = ''

   # 0 = Pe_nominal; 1 = Pe_hihi; 2 = Pe_hilo; 3 = Pe_lohi; 4 = Pe_lolo
   choices = [0,2,4] # pick from these randomly
   
   PeX = dict()
   PeY = dict()
   PeD = dict()

   for key1 in errors_dict.keys():
      for key2 in errors_dict[key1].keys():
         if errors_dict[key1][key2] is None:
            error_warning_string = error_warning_string + ' WARNING: nothing in errors_dict for ' + key1 + ' ' + key2 + '\n'
            continue
         else:
            x = errors_dict[key1][key2]['x']
            y = errors_dict[key1][key2]['y']
            d = errors_dict[key1][key2]['d']
            if calc is None or calc == 'mean' or calc == 'max':
               rnd = choices[random.randint(0,len(choices)-1)]
               #{{{
               if rnd == 0:
                  Pe1 = errors_dict[key1][key2]['Pe_aero']['nominal']
                  Pe2 = errors_dict[key1][key2]['Pe_gimp']['nominal']
               if rnd == 1:
                  Pe1 = errors_dict[key1][key2]['Pe_aero']['hihi']
                  Pe2 = errors_dict[key1][key2]['Pe_gimp']['hihi']
               if rnd == 2:
                  Pe1 = errors_dict[key1][key2]['Pe_aero']['hilo']
                  Pe2 = errors_dict[key1][key2]['Pe_gimp']['hilo']
               if rnd == 3:
                  Pe1 = errors_dict[key1][key2]['Pe_aero']['lohi']
                  Pe2 = errors_dict[key1][key2]['Pe_gimp']['lohi']
               if rnd == 4:
                  Pe1 = errors_dict[key1][key2]['Pe_aero']['lolo']
                  Pe2 = errors_dict[key1][key2]['Pe_gimp']['lolo']
               #}}}
               if '-' in key1:
                  ID = 'flowline' + key1.split('-')[1] + centerline[9:] + '_' + key2 + '_' + key1.split('-')[0]
               else:
                  ID = centerline + '_' + key2 + '_' + key1
               PeX[ID], PeY[ID], PeD[ID] = get_Pe3(ID, x, y, d, Pe1, Pe2, PeThreshold=PeThreshold)
            else:
               for rnd in choices:
               #{{{
                  if rnd == 0:
                     Pe1 = errors_dict[key1][key2]['Pe_aero']['nominal']
                     Pe2 = errors_dict[key1][key2]['Pe_gimp']['nominal']
                  if rnd == 1:
                     Pe1 = errors_dict[key1][key2]['Pe_aero']['hihi']
                     Pe2 = errors_dict[key1][key2]['Pe_gimp']['hihi']
                  if rnd == 2:
                     Pe1 = errors_dict[key1][key2]['Pe_aero']['hilo']
                     Pe2 = errors_dict[key1][key2]['Pe_gimp']['hilo']
                  if rnd == 3:
                     Pe1 = errors_dict[key1][key2]['Pe_aero']['lohi']
                     Pe2 = errors_dict[key1][key2]['Pe_gimp']['lohi']
                  if rnd == 4:
                     Pe1 = errors_dict[key1][key2]['Pe_aero']['lolo']
                     Pe2 = errors_dict[key1][key2]['Pe_gimp']['lolo']
               #}}}
                  if '-' in key1:
                     ID = 'flowline' + key1.split('-')[1] + centerline[9:] + '_' + key2 + '_' + key1.split('-')[0]
                  else:
                     ID = centerline + '_' + key2 + '_' + key1
                  _, _, PeDtmp = get_Pe3(ID, x, y, d, Pe1, Pe2, PeThreshold=PeThreshold)
                  ID = ID + '_choice{:d}'.format(rnd)
                  PeD[ID] = PeDtmp

   if calc is None or calc == 'mean' or calc == 'max':
      # Calculate stats
      statsDict = get_stats(PeX, PeY, PeD)
      
      # Remove outliers
      if remove_outliers:
         for outlier in statsDict['outliers']:
            PeD.pop(outlier, None)

         # New stats w/o outliers
         statsDict = get_stats(PeX, PeY, PeD)
      
      # Store the glacier-wide thinning limit for this combination
      gwtl = glacier_wide_thinning_limit(statsDict, calc=calc)
   else:
      # Find outliers
      q75, q25 = np.percentile(PeD.values(), [75 ,25])
      iqr = q75 - q25
      outlier_keys = [k for (k,v) in PeD.items() if v < q25-1.5*iqr or v > q75+1.5*iqr]
      if remove_outliers:
         for outlier in outlier_keys:
            PeD.pop(outlier, None)
      # Calculate glacier-wide thinning limit
      gwtl = np.max(PeD.values()) - np.std(PeD.values())

   return gwtl, error_warning_string
#}}}
