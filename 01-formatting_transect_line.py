from cmcrameri import cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os
import scipy.stats as stats

# Magnaprobe calibration value
lower_cal = 0.02  # m
upper_cal = 1.18  # m

# Line length in m
line_l = 200

# Point spacing in m
point_d = 1
plot_origin_flag = False

# WITH geopandas
#raw_file = '/mnt/data/UAF-data/raw_0/SALVO/20240417-ICE/armmagnaprobe-200m-20240417/salvo_arm_magnaprobe_200m_a1_geo1_20240417.csv'
#raw_file = '/mnt/data/UAF-data/raw_0/SALVO/20240418-BEO/beomagnaprobe/salvo_beo_magnaprobe_200m_a1_geo2_20240418.csv'
# With pyproj
#raw_file = '/mnt/data/UAF-data/raw_0/SALVO/20240421-ICE/icemagnaprobe-200m_roughness-20240421/salvo_ice_200m_a1_geodel_20240421.csv'
#raw_file = '/mnt/data/UAF-data/raw_0/SALVO/20240420-BEO/beomagnaprobe-200m_longline-20240420/salvo_beo_200m_a1_geodel_20240420.csv'
#raw_file = '/mnt/data/UAF-data/raw_0/SALVO/20240419-ARM/armmagnaprobe/salvo_arm_magnaprobe_200m_a1_geodel_20240419.csv'
raw_file = '/mnt/data/UAF-data/raw_0/SALVO/20240418-BEO/beomagnaprobe/salvo_beo_magnaprobe_200m_a1_geo2_20240418.csv'
raw_file = '/mnt/data/UAF-data/raw_0/SALVO/20240418-BEO/beomagnaprobe/salvo_beo_magnaprobe_200m_a1_geodel_20240418.csv'

# Generate output file
out_dir = os.path.dirname(raw_file)
input_fn = os.path.basename(raw_file)
if '_a' in input_fn:
    _increment_number = int(input_fn.split('_a')[-1].split('_')[0])
    _next_number = _increment_number + 1
    output_fn = input_fn.replace(str('a%.0f' % _increment_number), str('a%.0f' % _next_number))
    fig_fn = output_fn.replace(input_fn.split('.')[-1], 'pdf')
else:
    print('ERROR 001: double extension not defined')
fig_file = os.path.join(out_dir, fig_fn)
out_file = os.path.join(out_dir, output_fn)

# Read MagnaProbe raw data (*.dat) into a DataFrame
header_row = 0
date_cols = [1]
raw_df = pd.read_csv(raw_file, header=header_row, parse_dates=date_cols)

def euclidian_distance(x1, x2, y1, y2, z1=0, z2=0):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2)**2)

# Compute distance between two consecutive points
raw_df['distance'] = euclidian_distance(raw_df['x'], raw_df.shift()['x'], raw_df['y'], raw_df['y'].shift())
# Set distance at origin equal to 0
raw_df.iloc[0, raw_df.columns.get_loc('distance')] = 0
# Compute cumulative distance between two consecutive points
raw_df['linear distance'] = raw_df['distance'].cumsum()
# Compute distance between origin and given points
raw_df['distance from origin'] = euclidian_distance(raw_df['x'], raw_df.iloc[0].x, raw_df['y'], raw_df.iloc[0].y)

# Check if cleanup was correct
def quality_check(raw_df, lower_cal=lower_cal, upper_cal=upper_cal):
    # Set all Quality flag to 0
    raw_df['Quality'] = 0
    raw_df.loc[raw_df['SnowDepth'] < 0, 'Quality'] = 1

    # Look for lower and upper calibration value
    raw_df['Calibration'] = None
    raw_df.loc[raw_df['SnowDepth'] < lower_cal, 'Calibration'] = 'L'
    raw_df.loc[upper_cal < raw_df['SnowDepth'], 'Calibration'] = 'U'
    # Look for typical calibration pattern 'LU', 'UL', 'ULU', 'LUL', ...
    # TODO: to be implemented

    # If distance between consecutive points is smaller than a third of the median distance, the measurement may have been
    # taken at the same location
    d_median = raw_df['TrackDist'].median()
    raw_df.loc[raw_df['TrackDist'] < d_median / 3, 'Quality'] = 2

    # If time difference between two consecutive points is lower than 1 second, the second measurement could be a double
    # strike
    raw_df.loc[raw_df['Datetime'].diff() < dt.timedelta(0, 1), 'Quality'] = 3

    return raw_df

raw_df = quality_check(raw_df)
if len(raw_df) > 201:
    print('Too many points: N=%.0f' %len(raw_df))
elif len(raw_df) < 201:
    print('Too few points: N=%.0f' % len(raw_df))
else:
    raw_df['linear distance'] = np.arange(0, 201, 1)
if len(raw_df[raw_df.quality > 0]) > 0:
    print('Cleaning is still needed')

header_order = ['RECORD', 'datetime', 'Snow Depth', 'Latitude', 'Longitude', 'distance', 'linear distance', 'distance from origin', 'quality']
header_order = header_order + [col for col in raw_df.columns if col not in header_order]
raw_df = raw_df[header_order]
raw_df.to_csv(out_file, index=False)
print(output_fn)


# Plot basic figures:
# Title data
site = output_fn.split('_')[1]
date = output_fn.split('.')[0][-8:]
if 'longline' in output_fn:
    name = 'Long transect'
else:
    name = '200 m transect'

# Map data
if plot_origin_flag:
    raw_df['x0'] = raw_df['x'] - raw_df.iloc[0]['x']
    raw_df['y0'] = raw_df['y'] - raw_df.iloc[0]['y']
else:
    raw_df['x0'] = raw_df['x']
    raw_df['y0'] = raw_df['y']

snow_depth_scale = raw_df['SnowDepth'].max()-raw_df['SnowDepth'].min()
z0 = cm.davos(raw_df['SnowDepth']/snow_depth_scale)
# Figure
nrows, ncols = 2, 1
w_fig, h_fig = 8, 11
fig = plt.figure(figsize=[w_fig, h_fig])
gs1 = gridspec.GridSpec(nrows, ncols, height_ratios=[1, 1], width_ratios=[1])
ax = [[fig.add_subplot(gs1[0, 0])], [fig.add_subplot(gs1[1, 0])]]
ax = np.array(ax)
ax[0, 0].plot(raw_df['linear distance'], raw_df['SnowDepth'], color='steelblue')
ax[0, 0].fill_between(raw_df['linear distance'], raw_df['SnowDepth'], [0]*len(raw_df), color='lightsteelblue')

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

if len(raw_df) == 201:
    fig.savefig(fig_file)
    print(fig_file)
plt.show()
