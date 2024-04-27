from cmcrameri import cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os
import scipy.stats as stats

# Line length in m
line_l = 200

# Point spacing in m
point_d = 1
# WITH GEOPANDA
#raw_file = '/mnt/data/UAF-data/raw_0/SALVO/20240416-ARM/armmagnaprobe-longline-20240416/salvo_arm_magnaprobe_longline_a2_geo1_20240416.csv'
#raw_file = '/mnt/data/UAF-data/raw_0/SALVO/20240417-ICE/armmagnaprobe-200m-20240417/salvo_arm_magnaprobe_200m_a3_geo1_20240417.csv'
#raw_file = '/mnt/data/UAF-data/raw_0/SALVO/20240418-BEO/beomagnaprobe/salvo_beo_magnaprobe_200m_a1_geodel_20240418.csv'

# With pypro
raw_file = '/mnt/data/UAF-data/raw_0/SALVO/20240421-ICE/icemagnaprobe-200m_roughness-20240421/salvo_ice_200m_a2_geodel_20240421.csv'
raw_file = '/mnt/data/UAF-data/raw_0/SALVO/20240421-ICE/icemagnaprobe-200m_roughness-20240421/salvo_ice_roughness_a1_geodel_20240421.csv'
raw_file = '/mnt/data/UAF-data/raw_0/SALVO/20240420-BEO/beomagnaprobe-200m_longline-20240420/salvo_beo_longline_raw_geodel_20240420.dat'


def hs_stats_f(Y):
    hs_stats = {}
    hs_stats['avg'] = stats.tmean(Y)
    hs_stats['med'] = np.median(Y)
    hs_stats['max'] = stats.tmax(Y)
    hs_stats['maxloc'] = np.where(Y == hs_stats['max'])[0]
    hs_stats['min'] = stats.tmin(Y)
    hs_stats['minloc'] = np.where(Y == hs_stats['min'])[0]
    hs_stats['sdev'] = stats.tstd(Y)  # Standard deviation
    hs_stats['adev'] = np.sum(np.abs((Y - np.mean(Y))))/len(Y)
    hs_stats['sem'] = stats.tsem(Y)  # Standard error of mean
    hs_stats['rms'] = np.sqrt(np.sum(Y**2)/len(Y))  # Root mean square
    hs_stats['skew'] = stats.skew(Y, bias=False)
    hs_stats['kurt'] = stats.kurtosis(Y, bias=False)
    return hs_stats

hs_stats_f(raw_df['SnowDepth'])
hs_stats_f(raw_df['TrackDist'])


# Generate output file
out_dir = os.path.dirname(raw_file)
input_fn = os.path.basename(raw_file)
fig_fn = input_fn.replace(input_fn.split('.')[-1], 'pdf')
fig_file = os.path.join(out_dir, fig_fn)

# Read MagnaProbe raw data (*.dat) into a DataFrame
header_row = 0
date_cols = [1]
raw_df = pd.read_csv(raw_file, header=header_row,parse_dates=date_cols)


# Plot basic figures:
site = fig_fn.split('_')[1]
date = fig_fn.split('.')[0][-8:]
if 'longline' in fig_fn:
    name = 'Long transect'
elif '200' in fig_fn:
    name = '200 m transect'
elif 'roughness' in fig_fn:
    name = 'roughness patches'
# Map data
raw_df['x0'] = raw_df['x'] - raw_df.iloc[0]['x']
raw_df['y0'] = raw_df['y'] - raw_df.iloc[0]['y']
snow_depth_scale = 1.2
z0 = cm.davos(raw_df['Snow Depth']/snow_depth_scale)

# Figure
nrows, ncols = 2, 1
w_fig, h_fig = 8, 11
fig = plt.figure(figsize=[w_fig, h_fig])
gs1 = gridspec.GridSpec(nrows, ncols, height_ratios=[1, 1], width_ratios=[1])
ax = [[fig.add_subplot(gs1[0, 0])], [fig.add_subplot(gs1[1, 0])]]
ax = np.array(ax)
ax[0, 0].plot(raw_df['linear distance'], raw_df['Snow Depth'], color='steelblue')
ax[0, 0].fill_between(raw_df['linear distance'], raw_df['Snow Depth'], [0]*len(raw_df), color='lightsteelblue')

ax[0, 0].set_xlabel('Distance along the transect (m)')
ax[0, 0].set_ylabel('Snow depth (m)')
ax[0, 0].set_xlim([0, raw_df['linear distance'].max()])
ax[0, 0].set_ylim([0, 1.2])

ax[1, 0].scatter(raw_df['x0'], raw_df['y0'], c=z0)
ax[1, 0].set_xlabel('X distance from origin (m)')
ax[1, 0].set_ylabel('Y distance from origin (m)')

if np.diff(ax[1, 0].get_ylim()) < 200:
    y_min = min(ax[1, 0].get_ylim())
    y_max = max(ax[1, 0].get_ylim())
    y_avg = np.mean([y_min, y_max])
    ax[1, 0].set_ylim([y_avg-100, y_avg+100])

fig.suptitle((' - '.join([site.upper(), name, date])))
plt.show()

fig.savefig(fig_file)
print(fig_file)