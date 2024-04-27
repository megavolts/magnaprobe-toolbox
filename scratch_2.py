from cmcrameri import cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os

raw_file = '/mnt/data/UAF-data/raw_0/SALVO/20240421-ICE/icemagnaprobe-200m_roughness-20240421/salvo_ice_roughness_a1_geodel_20240421.csv'
raw_file = '/mnt/data/UAF-data/raw_0/SALVO/20240420-BEO/beomagnaprobe-200m_longline-20240420/salvo_beo_longline_a1_geodel_20240420.csv'
raw_file = '/mnt/data/UAF-data/raw_0/SALVO/20240418-BEO/beomagnaprobe/salvo_beo_magnaprobe_longline_a1_geo2_20240418.csv'
raw_file = '/mnt/data/UAF-data/raw_0/SALVO/20240417-ICE/icemagnaprobe-200m-20240417/salvo_ice_magnaprobe_200m_a1_geo1_20240417.csv'
raw_file = '/mnt/data/UAF-data/raw_0/SALVO/20240416-ARM/armmagnaprobe-longline-20240416/salvo_arm_magnaprobe_longline_a2_geo1_20240416.csv'

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

# Find best fit for x0, y0
from sklearn.linear_model import LinearRegression
raw_df['x0'] = raw_df['x'] - raw_df.iloc[0]['x']
raw_df['y0'] = raw_df['y'] - raw_df.iloc[0]['y']
X = raw_df[["x0"]]
Y = raw_df[["y0"]]

regressor = LinearRegression()
regressor.fit(X, Y)
y_pred = regressor.predict(X)
#regressor.coef_
#regressor.intercept_
angle = np.arctan(regressor.coef_)
x0 = 0
y0 = regressor.intercept_

# Rotate
def rotate(x0, y0, x, y, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    qx = x0 + np.cos(angle) * (x - x0) - np.sin(angle) * (y - y0)
    qy = y0 + np.sin(angle) * (x - x0) + np.cos(angle) * (y - y0)
    return qx, qy

raw_df['xr'], raw_df['yr'] = rotate(x0, y0[0], raw_df['x0'], raw_df['y0'], -regressor.coef_[0])

# Check best fit
XR = raw_df[["xr"]]
YR = raw_df[["yr"]]
regressor = LinearRegression()
regressor.fit(XR, YR)
YR_pred = regressor.predict(XR)

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
#raw_df['x0'] = raw_df['x'] - raw_df.iloc[0]['x']
#raw_df['y0'] = raw_df['y'] - raw_df.iloc[0]['y']
#raw_df['x0'] = raw_df['x'] - raw_df.iloc[0]['x']
#raw_df['y0'] = raw_df['y'] - raw_df.iloc[0]['y']
X = raw_df['xr']
Y = raw_df['yr']
snow_depth_scale = 1.2
Z = cm.davos(raw_df['Snow Depth']/snow_depth_scale)

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

ax[1, 0].scatter(X, Y, c=Z)
ax[1, 0].plot(XR, YR_pred, color='r')

ax[1, 0].set_xlabel('X distance from origin (m)')
ax[1, 0].set_ylabel('Y distance from origin (m)')

if np.diff(ax[1, 0].get_ylim()) < 50:
    y_min = min(ax[1, 0].get_ylim())
    y_max = max(ax[1, 0].get_ylim())
    y_avg = np.mean([y_min, y_max])
    ax[1, 0].set_ylim([y_avg-25, y_avg+25])

ax[1, 0].set_xlim([min(X), max(X)])
ax[1, 0].text(200, 50, str("rotate by %.2f°" % np.rad2deg(-angle[0][0])))
ax[1, 0].text(200, 40, str("max dev by %.1f m @ " % raw_df['yr'].abs().max()))
ax[1, 0].scatter(raw_df.loc[np.isclose(raw_df['yr'] - raw_df['yr'].abs().max(), 0), 'xr'], raw_df['yr'].abs().max(), c='r')
fig.suptitle((' - '.join([site.upper(), name, date])))
plt.show()

input_fn = os.path.basename(raw_file)
fig_fn = input_fn.replace(input_fn.split('.')[-1], 'pdf')
fig_fn = fig_fn.replace('.pdf', '-rot.pdf')
fig_file = os.path.join(out_dir, fig_fn)
fig.savefig(fig_file)


