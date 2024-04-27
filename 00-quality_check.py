from cmcrameri import cm
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import gridspec as gridspec
import numpy as np
import os
import pandas as pd
import pyproj
import scipy.stats as stats

# -- USER VARIABLES
# ESPG code for local projection
local_ESPG = '3338'  # for Alaska
# Magnaprobe calibration value
lower_cal = 0.02  # m
upper_cal = 1.18  # m
# Headers
# Header to rename
rename_header = {'TIMESTAMP': 'Datetime'}
# Header to remove
drop_header = ['RECORD', 'BattVolts', 'latitude_a', 'latitude_b', 'Longitude_a', 'Longitude_b', 'fix_quality',
               'nmbr_satellites', 'HDOP', 'altitudeB', 'DepthVolts', 'LatitudeDDDDD', 'LongitudeDDDDD', 'month',
               'dayofmonth', 'hourofday', 'minutes', 'seconds', 'microseconds']

# Flag
# Enable plotting with relative coordinate to the transect origin x0=0, y0=0
plot_origin_flag = False

# Processed with pyproj
#raw_file = '/mnt/data/UAF-data/raw_0/SALVO/20240418-BEO/beomagnaprobe/salvo_beo_magnaprobe_200m_raw_geo2_20240418.dat'
raw_file = '/mnt/data/UAF-data/raw_0/SALVO/20240421-ICE/icemagnaprobe-200m_roughness-20240421/salvo_ice_200m_raw_geodel_20240421.dat'
#raw_file = '/mnt/data/UAF-data/raw_0/SALVO/20240421-ICE/icemagnaprobe-200m_roughness-20240421/salvo_ice_roughness_raw_geodel_20240421.dat'
#raw_file = '/mnt/data/UAF-data/raw_0/SALVO/20240420-BEO/beomagnaprobe-200m_longline-20240420/salvo_beo_200m_raw_geodel_20240420.raw.dat'
#raw_file = '/mnt/data/UAF-data/raw_0/SALVO/20240420-BEO/beomagnaprobe-200m_longline-20240420/salvo_beo_longline_raw_geodel_20240420.dat'
#raw_file = '/mnt/data/UAF-data/raw/SALVO/20240419-ARM/armmagnaprobe/salvo_arm_magnaprobe_200m_raw_geodel_20240419.dat'
#raw_file = '/mnt/data/UAF-data/raw_0/SALVO/20240418-BEO/beomagnaprobe/salvo_beo_magnaprobe_longline_raw_geo2_20240418.dat'
#raw_file = '/mnt/data/UAF-data/raw_0/SALVO/20240418-BEO/beomagnaprobe/salvo_beo_magnaprobe_200m_raw_geo2_20240418.dat'
#raw_file = '/mnt/data/UAF-data/raw/SALVO/20240418-BEO/beomagnaprobe/salvo_beo_magnaprobe_200m_raw_geodel_20240418.dat'
#raw_file = '/mnt/data/UAF-data/raw/SALVO/20240417-ICE/icemagnaprobe-200m-20240417/salvo_ice_magnaprobe_200m_raw_geo1_20240417.dat'
raw_file = '/mnt/data/UAF-data/raw/SALVO/20240416-ARM/magnaprobe/salvo_arm_longline_magnaprobe-geo1_20240416.raw.dat'

# Generate output file
base_dir = os.path.dirname(raw_file)
out_dir = base_dir.replace('/raw/', '/raw_0/')

# Input filename
input_fn = os.path.basename(raw_file)
if '.raw.' in input_fn:
    output_fn = input_fn.replace('.raw.', '.a0.')
elif '.a' in input_fn:
    _increment_number = int(input_fn.split('.')[-2][1:])
    _next_number = _increment_number + 1
    output_fn = input_fn.replace(str('a%.0f' % _increment_number), str('a%.0f' % _next_number))
out_file = os.path.join(out_dir, output_fn)
out_dir = ('/').join(out_file.split('/')[:-1])
if not os.path.exists(out_dir):
    os.makedirs(('/').join(out_file.split('/')[:-1]))

# Read MagnaProbe raw data (*.dat) into a DataFrame
header_row = 1
raw_df = pd.read_csv(raw_file, header=header_row)

# Drop junk header
raw_df = raw_df.drop(raw_df.index[:2])

# parse timesteamp
raw_df = raw_df.rename(columns=rename_header)
try:
    raw_df['Datetime'] = pd.to_datetime(raw_df['Datetime'], format='ISO8601')
except ValueError:
    # create artificial timestamp starting at midnight local time
    date = pd.to_datetime(out_file.split('.')[0][-8:])
    raw_df['Datetime'] = pd.date_range(date, periods=len(raw_df), freq="s")

# coordinate columns:
raw_df['Latitude'] = raw_df['latitude_a'].astype('float') + raw_df['LatitudeDDDDD'].astype('float')/60
raw_df['Longitude'] = raw_df['Longitude_a'].astype('float') + raw_df['LongitudeDDDDD'].astype('float')/60

# depth in m not cm
raw_df['SnowDepth'] = raw_df['DepthCm'].astype('float') / 100

# Create X,Y geospatial in ESPG 3338 reference for Alaska
# Convert from ESPG 4326 (WSG84) to ESPG 3338
xform = pyproj.Transformer.from_crs('4326', local_ESPG)
raw_df['x'], raw_df['y'] = xform.transform(raw_df['Latitude'], raw_df['Longitude'])
raw_df['z'] = [0]*len(raw_df)

def euclidian_distance(x1, x2, y1, y2, z1=0, z2=0):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2)**2)

# Compute distance between two consecutive points
raw_df['TrackDist'] = euclidian_distance(raw_df['x'], raw_df['x'].shift(), raw_df['y'], raw_df['y'].shift())
raw_df.iloc[0, raw_df.columns.get_loc('TrackDist')] = 0
# Compute cumulative sum of distance between two consecutive points
raw_df['TrackDistCum'] = raw_df['TrackDist'].cumsum()
raw_df.iloc[0, raw_df.columns.get_loc('TrackDistCum')] = 0
# Compute distance between the origin point and the current point
raw_df['DistOrigin'] = euclidian_distance(raw_df['x'], raw_df.iloc[0]['x'], raw_df['y'], raw_df.iloc[0]['y'])

# Quality check
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

# Reorganize headers
raw_df = raw_df[[col for col in raw_df.columns if col not in drop_header]]

header_order = ['Counter', 'Datetime', 'SnowDepth', 'Latitude', 'Longitude', 'TrackDist', 'TrackDistCum', 'DistOrigin', 'Quality']
header_order = header_order + [col for col in raw_df.columns if col not in header_order]
raw_df = raw_df[header_order]

raw_df.to_csv(out_file, index=False)
print(output_fn)

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

# Data status plot
# Move origin points to x0, y0
if plot_origin_flag:
    raw_df['x0'] = raw_df['x'] - raw_df.iloc[0]['x']
    raw_df['y0'] = raw_df['y'] - raw_df.iloc[0]['y']
else:
    raw_df['x0'] = raw_df['x']
    raw_df['y0'] = raw_df['y']

snow_depth_scale = 1.2
z0 = cm.davos(raw_df['SnowDepth']/snow_depth_scale)

# Define figure subplots
nrows, ncols = 2, 1
w_fig, h_fig = 8, 11
fig = plt.figure(figsize=[w_fig, h_fig])
gs1 = gridspec.GridSpec(nrows, ncols, height_ratios=[1, 1], width_ratios=[1])
ax = [[fig.add_subplot(gs1[0, 0])], [fig.add_subplot(gs1[1, 0])]]
ax = np.array(ax)
ax[0, 0].plot(raw_df['TrackDistCum'], raw_df['SnowDepth'], color='steelblue')
ax[0, 0].fill_between(raw_df['TrackDistCum'], raw_df['SnowDepth'], [0]*len(raw_df), color='lightsteelblue')

ax[0, 0].set_xlabel('Distance along the transect (m)')
ax[0, 0].set_ylabel('Snow Depth (m)')
ax[0, 0].set_xlim([0, raw_df['TrackDistCum'].max()])
ax[0, 0].set_ylim([0, 1.2])

ax[1, 0].scatter(raw_df['x0'], raw_df['y0'], c=z0)
ax[1, 0].set_xlabel('X coordinate (m)')
ax[1, 0].set_ylabel('Y coordinate (m)')

if np.diff(ax[1, 0].get_ylim()) < 200:
    y_min = min(ax[1, 0].get_ylim())
    y_max = max(ax[1, 0].get_ylim())
    y_avg = np.mean([y_min, y_max])
    ax[1, 0].set_ylim([y_avg-100, y_avg+100])
plt.show()
