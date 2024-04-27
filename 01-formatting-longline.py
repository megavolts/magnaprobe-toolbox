from cmcrameri import cm
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os
import scipy.stats as stats


# Magnaprobe calibration value
lower_cal = 0.02  # m
upper_cal = 1.18  # m


plot_origin_flag = False

raw_file = '/mnt/data/UAF-data/raw_0/SALVO/20240416-ARM/magnaprobe/salvo_arm_longline_magnaprobe-geo1_20240416.a1.csv'

# Generate output file
out_dir = os.path.dirname(raw_file)
input_fn = os.path.basename(raw_file)
if '.a' in input_fn:
    _increment_number = int(input_fn.split('.a')[-1].split('.')[0])
    _next_number = _increment_number + 1
    output_fn = input_fn.replace(str('a%.0f' % _increment_number), str('a%.0f' % _next_number))
    fig_fn = output_fn.replace(input_fn.split('.')[-1], 'pdf')
else:
    print('ERROR 001: double extension not defined')
fig_file = os.path.join(out_dir, fig_fn)
out_file = os.path.join(out_dir, output_fn)

# Read MagnaProbe QC data (*.csv) into a DataFrame
header_row = 0
date_cols = [1]
raw_df = pd.read_csv(raw_file, header=header_row, parse_dates=date_cols)

def euclidian_distance(x1, x2, y1, y2, z1=0, z2=0):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2)**2)

# Compute distance between two consecutive points to account for removed points
raw_df['TrackDist'] = euclidian_distance(raw_df['x'], raw_df.shift()['x'], raw_df['y'], raw_df['y'].shift())
# Set distance at origin equal to 0
raw_df.iloc[0, raw_df.columns.get_loc('TrackDist')] = 0
# Compute cumulative distance between two consecutive points
raw_df['TrackDistCum'] = raw_df['TrackDist'].cumsum()
raw_df.iloc[0, raw_df.columns.get_loc('TrackDistCum')] = 0
# Compute distance between origin and given points
raw_df['DistOrigin'] = euclidian_distance(raw_df['x'], raw_df.iloc[0].x, raw_df['y'], raw_df.iloc[0].y)

# cleanup check

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

# Inspired by https://www.wavemetrics.net/doc/igorman/III-07%20Analysis.pdf
def wavestats(Y):
    hs_stats = {}
    hs_stats['sum'] = np.sum(Y)
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

raw_df = quality_check(raw_df)

if len(raw_df[raw_df.Quality > 0]) > 0:
    print('Cleaning is still needed')

# export QC, cleaned data file
raw_df.to_csv(out_file, index=False)
print(output_fn)

# Compute wave statistic for snow depth
hs_stat = wavestats(raw_df['SnowDepth'])

# Compute snow depth histogram
hs_max = 1.2
hs_min = 0
d_hs = 0.1
n_bins = int(np.round((hs_max - hs_min) / d_hs + 1))
counts, bins = np.histogram(raw_df['SnowDepth'], np.arange(hs_min, hs_max+d_hs/2, d_hs), density=True)
hs_bin_mid = bins[:-1] + np.diff(bins)/2

# For log normal to snow depth distribution
pdf_p = stats.lognorm.fit(raw_df['SnowDepth'])
hs_x = np.arange(hs_min, hs_max+0.001, 0.001)
hs_fitted_ln = stats.lognorm(*pdf_p)

# TODO: semivariograph. Waiting for MS answer

# Plot basic figures:
site = output_fn.split('_')[1]
date = output_fn.split('.')[0][-8:]
if 'longline' in output_fn:
    name = 'Long transect'
elif '200' in output_fn:
    name = '200 m transect'
elif 'library' in output_fn:
    name = 'Library site'

# Data status plot
# Move origin points to x0, y0
if plot_origin_flag:
    raw_df['x0'] = raw_df['x'] - raw_df.iloc[0]['x']
    raw_df['y0'] = raw_df['y'] - raw_df.iloc[0]['y']
else:
    raw_df['x0'] = raw_df['x']
    raw_df['y0'] = raw_df['y']
snow_depth_scale = raw_df['SnowDepth'].max()-raw_df['SnowDepth'].min()
z0 = cm.davos(raw_df['SnowDepth']/snow_depth_scale)
plt.style.use('ggplot')

# Define figure subplots
nrows, ncols = 4, 1
w_fig, h_fig = 8, 11
fig = plt.figure(figsize=[w_fig, h_fig])
gs1 = gridspec.GridSpec(nrows, ncols, height_ratios=[1, 1, 1, 1], width_ratios=[1])
ax = [[fig.add_subplot(gs1[0, 0])], [fig.add_subplot(gs1[1, 0])], [fig.add_subplot(gs1[2, 0])], [fig.add_subplot(gs1[3, 0])]]
ax = np.array(ax)
ax[0, 0].plot(raw_df['TrackDistCum'], raw_df['SnowDepth'], color='steelblue')
ax[0, 0].fill_between(raw_df['TrackDistCum'], raw_df['SnowDepth'], [0]*len(raw_df), color='lightsteelblue')

ax[0, 0].set_xlabel('Distance along the transect (m)')
ax[0, 0].set_ylabel('Snow Depth (m)')
ax[0, 0].set_xlim([0, raw_df['TrackDistCum'].max()])
ax[0, 0].set_ylim([0, 1.2])

ax[1, 0].scatter(raw_df['x0'], raw_df['y0'], c=z0)
if plot_origin_flag:
    ax[1, 0].set_xlabel('X coordinate (m)')
    ax[1, 0].set_ylabel('Y coordinate (m)')
else:
    ax[1, 0].set_xlabel('UTM X coordinate (m)')
    ax[1, 0].set_ylabel('UTM Y coordinate (m)')

# Snow depth probability function:
ax[2, 0].bar(hs_bin_mid, counts, 0.1, edgecolor='steelblue', color='lightsteelblue')
ax[2, 0].plot(hs_x , hs_fitted_ln.pdf(hs_x), 'k')
ax[2, 0].set_ylabel('PDF')
ax[2, 0].set_xlabel('Snow Depth (m)')

# ax[2, 0].text[0.6, 3 ]
if np.diff(ax[1, 0].get_ylim()) < 200:
    y_min = min(ax[1, 0].get_ylim())
    y_max = max(ax[1, 0].get_ylim())
    y_avg = np.mean([y_min, y_max])
    ax[1, 0].set_ylim([y_avg-100, y_avg+100])
fig.suptitle((' - '.join([site.upper(), name, date])))
plt.show()
