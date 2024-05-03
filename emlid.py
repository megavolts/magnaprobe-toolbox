import os

import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib import gridspec as gridspec
import numpy as np
import math
import pyproj

# This file convert all time to UTC

# in local time, AKDT
survey_tz = -8  # AKTD
survey_dt = 0  # in seconds
gps_tz = 0  # UTC

# Define CSV, POS, and output file locations
# BEO 20240420
# survey_fp = '/mnt/data/UAF-data/working_a/SALVO/20240420-BEO/magnaprobe/salvo_beo_longline_magnaprobe-geodel_20240420.a3.csv'
# survey_fp = '/mnt/data/UAF-data/working_a/SALVO/20240420-BEO/magnaprobe/salvo_beo_line_magnaprobe-geodel_20240420.a3.csv'
# ppk_pos_fp = '/mnt/data/UAF-data/working_a/SALVO/20240420-BEO/emlid/salvo_beo_line-longline_reachm2-rinex-2125_20240420.00/reachm2_raw_202404202125.pos'
# out_csv_fn = '/mnt/data/UAF-data/working_b/SALVO/20240420-BEO/beomagnaprobe-200m_longline-20240420/salvo_beo_longline_a2_geodel_20240420.csv'

# ICE 20240421
survey_fp = '/mnt/data/UAF-data/working_a/SALVO/20240421-ICE/magnaprobe/salvo_ice_line_magnaprobe-geodel_20240421.a5.csv'
ppk_pos_fp = '/mnt/data/UAF-data/working_a/SALVO/20240421-ICE/emlid/salvo_ice_line_reachm2-rinex-2001_20240421.00.zip/reachm2_raw_202404212001.pos'
out_csv_fn = '/mnt/data/UAF-data/working_b/SALVO/20240420-BEO/beomagnaprobe-200m_longline-20240420/salvo_beo_longline_a2_geodel_20240420.csv'

# ARM
survey_fp = '/mnt/data/UAF-data/working_a/SALVO/20240419-ARM/magnaprobe/salvo_arm_line_magnaprobe-geodel_20240419.a3.csv'
ppk_pos_fp = '/mnt/data/UAF-data/working_a/SALVO/20240419-ARM/emlid/salvo_arm_line_reachm2-rinex-1852_20240419.00/reachm2_raw_202404191852.pos'
out_csv_fn = '/mnt/data/UAF-data/working_b/SALVO/20240420-BEO/beomagnaprobe-200m_longline-20240420/salvo_beo_longline_a2_geodel_20240420.csv'


# Load the Files
print('Loading: %s' % survey_fp)
survey_df = pd.read_csv(survey_fp, header=0)
# Convert to UTC
survey_df['Timestamp'] = pd.to_datetime(survey_df['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f') - timedelta(
    hours=survey_tz) + timedelta(seconds=survey_dt)
# Round to nearest 200ms
rounding_factor = pd.to_timedelta(1000, unit='ms')  # Define rounding to the nearest 200 milliseconds
survey_df['Timestamp'] = survey_df['Timestamp'].apply(lambda x :x.round(rounding_factor))

start_time = pd.to_datetime(survey_df['Timestamp'].min()) - timedelta(0, 60)
end_time = pd.to_datetime(survey_df['Timestamp'].max()) + timedelta(0, 60)

# PPK Data
header = 'Date Time Latitude Longitude Altitude   Q  ns   sdn(m)   sde(m)   sdu(m)  sdne(m)  sdeu(m)  sdun(m) age(s)  ratio'
print('Loading: %s' % ppk_pos_fp)
ppk_pos_df = pd.read_csv(ppk_pos_fp, comment='%', sep='\s+', names=header.split())
ppk_pos_df['Datetime'] = pd.to_datetime(ppk_pos_df['Date'] + ' ' + ppk_pos_df['Time'], format='%Y/%m/%d %H:%M:%S.%f')

# Convert ppk_pos_df into X, Y and Z:
local_EPSG = 3338
xform = pyproj.Transformer.from_crs('4326', local_EPSG)
ppk_pos_df['X'], ppk_pos_df['Y'] = xform.transform(ppk_pos_df['Latitude'], ppk_pos_df['Longitude'])
ppk_pos_df['Z'] = ppk_pos_df['Altitude']
import magnaprobe_toolbox as mt
ppk_pos_df = mt.tools.compute_distance(ppk_pos_df)

ppk_pos_df = ppk_pos_df[(start_time < ppk_pos_df['Datetime']) & (ppk_pos_df['Datetime'] < end_time)]

# Plot Elevation profile

### Format the dataFrames
## Round the MagnaProbe timestamp to match nearest 5Hz rate
## (Incorrectly) Assuming that MagnaProbe Realtime() 'TIMESTAMP' is syncronized with the GPS NMEA string
## This can be adjusted based on how far off the MagnaProbe clock is from actual UTC. The 18 second timedelta is for UTC -> GPST and should stay
## Format the Emlid pos file and convert from GPST to UTC
ppk_pos_df['Timestamp'] = pd.to_datetime(ppk_pos_df['Datetime'], format='%Y-%m-%d %H:%M:%S')

outDF = pd.merge(survey_df, ppk_pos_df, on='Timestamp', how='left')

##Format Output to match original magnaProbe format
# Convert DD to DDM and isolate the decimal portion
# outDF['latitude_a'] = outDF['latitude(deg)'].apply(lambda x: math.modf(x)[1])
# outDF['latitude_b'] = outDF['latitude(deg)'].apply(lambda x: math.modf(x)[0]).multiply(60)
# outDF['longitude_a'] = outDF['longitude(deg)'].apply(lambda x: math.modf(x)[1])
# outDF['longitude_b'] = outDF['longitude(deg)'].apply(lambda x: math.modf(x)[0]).multiply(60)
# outDF['latitudeDDDDD'] = outDF['latitude(deg)'].apply(lambda x: math.modf(x)[0])
# outDF['longitudeDDDDD'] = outDF['longitude(deg)'].apply(lambda x: math.modf(x)[0])
# Isolate the timestamps
# outDF['month'] = [d.date().month for d in outDF['DateTime']]
# outDF['dayofmonth'] = [d.date().day for d in outDF['DateTime']]
# outDF['hourofday'] = [d.time().hour for d in outDF['DateTime']]
# outDF['minutes'] = [d.time().minute for d in outDF['DateTime']]
# outDF['seconds'] = [d.time().second for d in outDF['DateTime']]
# outDF['microseconds'] = [d.time().microsecond for d in outDF['DateTime']]
# Create df to output
# # out_df = outDF[
#     ['TIMESTAMP', 'RECORD', 'Counter', 'DepthCm', 'BattVolts', 'latitude_a', 'latitude_b', 'longitude_a', 'longitude_b',
#      'Q', 'ns', 'height(m)', 'DepthVolts', 'latitudeDDDDD', 'longitudeDDDDD', 'month', 'dayofmonth', 'hourofday',
#      'minutes', 'seconds', 'microseconds', 'latitude_emlid', 'longitude(deg)', 'sdn(m)', 'sde(m)', 'sdu(m)', 'sdne(m)']]
# out_df = out_df.rename(columns={'Q': 'fix_quality', 'ns': 'nmbr_satellites', 'height(m)': 'altitudeB'})
out_df = outDF
import matplotlib.pyplot as plt
import seaborn as sns

# create boxplot with a different y scale for different rows
labels = ["Fix", "Float", "Single"]
groups = outDF.groupby('Q')
fig, axes = plt.subplots(1, len(labels))
i = 0
for ID, group in groups:
    print(ID, group)
    ax = sns.boxplot(y=group['sdne(m)'].abs(), ax=axes.flatten()[i])
    ax.set_xlabel(labels[i])
    ax.set_ylim(group['sdne(m)'].abs().min() * 0.85, group['sdne(m)'].abs().max() * 1.15)
    if i == 0:
        ax.set_ylabel('Location Uncertainty (m)')
    else:
        ax.set_ylabel('')
    ax.text(0.1, 0.942, "N= " + str(len(group)), transform=ax.transAxes, size=10, weight='bold')
    ax.text(0.1, 0.9, "Mean= " + str(3 * group['sdne(m)'].abs().mean()) + " m", transform=ax.transAxes, size=10,
            weight='bold')
    i += 1
fig.suptitle('Emlid GPS Location Precision', fontweight='bold')
fig.text(0.5, 0.02, 'GPS Location Solution', ha='center', va='center')
plt.subplots_adjust(left=0.1,
                    right=0.9,
                    wspace=0.4,
                    hspace=0.4)
plt.show()
##Write out new file
print("Writing out: %s" % out_csv_fn)
import os
if not os.path.exists(os.path.dirname(out_csv_fn)):
    os.makedirs(os.path.dirname(out_csv_fn))
out_df.to_csv(out_csv_fn, index=False)
out_df.to_feather(out_csv_fn.split(".")[0]+".feather")


ncols = 1
nrows = 4
plt.figure()
w_fig, h_fig = 8, 11
fig = plt.figure(figsize=[w_fig, h_fig])
gs1 = gridspec.GridSpec(4, ncols, height_ratios=[1] * 4, width_ratios=[1])
ax = [[fig.add_subplot(gs1[0, 0])], [fig.add_subplot(gs1[1, 0])], [fig.add_subplot(gs1[2, 0])],
      [fig.add_subplot(gs1[3, 0])]]
ax = np.array(ax)


ax[0, 0].plot(outDF['DistOrigin_y'], outDF['Z_y'])

ax[0, 0].plot(outDF.loc[outDF.Q == 1, 'DistOrigin_y'], outDF.loc[outDF.Q == 1, 'Z_y'], color='green', marker='o', linestyle='')
ax[0, 0].plot(outDF.loc[outDF.Q == 5, 'DistOrigin_y'], outDF.loc[outDF.Q == 5, 'Z_y'], color='red', marker='x', linestyle='')

#ax[0, 0].set_ylim([0, 10])
ax[1, 0].plot(ppk_pos_df.loc[ppk_pos_df.Q < 5, 'DistOrigin'], ppk_pos_df.loc[ppk_pos_df.Q < 5, 'Z'], 'k', alpha=0.6)
#ax[1, 0].set_ylim([0, 10])
ax[0, 0].set_xlabel('Distance (m)')
ax[0, 0].set_xlabel('Elevation (m)')
ax[2, 0].plot(outDF['X_y'], outDF['Y_y'], 'ro')
ax[2, 0].plot(outDF['X_x'], outDF['Y_x'], 'xk')
ax[3, 0].plot(outDF['DistOrigin_y'], outDF['DistOrigin_x'])
ax[3, 0].set_xlabel('Distance (m), emlid')
ax[3, 0].set_xlabel('Distance (m), magna')


plt.show()