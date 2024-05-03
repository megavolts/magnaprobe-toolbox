import pandas as pd
from datetime import timedelta
import math

# Define CSV, POS, and output file locations
survey_pts_csv_fn = "/mnt/data/UAF-data/raw_0/SALVO/20240420-BEO/beomagnaprobe-200m_longline-20240420/salvo_beo_longline_a2_geodel_20240420.csv"
ppk_pos_fn = "/mnt/data/UAF-data/working_a/SALVO/20240420-BEO/emlid/salvo_beo_line-longline_reachm2-rinex-2125_20240420.00/reachm2_raw_202404202125.pos"
out_csv_fn = '/mnt/data/UAF-data/working_b/SALVO/20240420-BEO/beomagnaprobe-200m_longline-20240420/salvo_beo_longline_a2_geodel_20240420.csv'
# Load the Files
print('Loading: %s' % survey_pts_csv_fn)
survey_pts = pd.read_csv(survey_pts_csv_fn, header=0)
header = 'Date GPST latitude(deg) longitude(deg)  height(m)   Q  ns   sdn(m)   sde(m)   sdu(m)  sdne(m)  sdeu(m)  sdun(m) age(s)  ratio'
print('Loading: %s' % ppk_pos_fn)
ppk_pos = pd.read_csv(ppk_pos_fn, comment='%', delim_whitespace=True, names=header.split(), parse_dates=[[0, 1]])

### Format the dataFrames
## Round the MagnaProbe timestamp to match nearest 5Hz rate
## (Incorrectly) Assuming that MagnaProbe Realtime() 'TIMESTAMP' is syncronized with the GPS NMEA string
## This can be adjusted based on how far off the MagnaProbe clock is from actual UTC. The 18 second timedelta is for UTC -> GPST and should stay
survey_pts['TIMESTAMP2'] = pd.to_datetime(survey_pts['datetime'], format='%Y-%m-%d %H:%M:%S.%f') + timedelta(
    hours=9) - timedelta(seconds=18)
rounding_factor = pd.to_timedelta(200, unit='ms')  # Define rounding to the nearest 200 milliseconds
survey_pts['DateTime'] = survey_pts['TIMESTAMP2'].round(rounding_factor)
## Format the Emlid pos file and convert from GPST to UTC
ppk_pos['DateTime'] = pd.to_datetime(ppk_pos['Date_GPST'], format='%Y-%m-%d %H:%M:%S.%f')
outDF = pd.merge(survey_pts, ppk_pos, on='DateTime', how='left')

##Format Output to match original magnaProbe format
# Convert DD to DDM and isolate the decimal portion
outDF['latitude_a'] = outDF['latitude(deg)'].apply(lambda x: math.modf(x)[1])
outDF['latitude_b'] = outDF['latitude(deg)'].apply(lambda x: math.modf(x)[0]).multiply(60)
outDF['longitude_a'] = outDF['longitude(deg)'].apply(lambda x: math.modf(x)[1])
outDF['longitude_b'] = outDF['longitude(deg)'].apply(lambda x: math.modf(x)[0]).multiply(60)
outDF['latitudeDDDDD'] = outDF['latitude(deg)'].apply(lambda x: math.modf(x)[0])
outDF['longitudeDDDDD'] = outDF['longitude(deg)'].apply(lambda x: math.modf(x)[0])
# Isolate the timestamps
outDF['month'] = [d.date().month for d in outDF['DateTime']]
outDF['dayofmonth'] = [d.date().day for d in outDF['DateTime']]
outDF['hourofday'] = [d.time().hour for d in outDF['DateTime']]
outDF['minutes'] = [d.time().minute for d in outDF['DateTime']]
outDF['seconds'] = [d.time().second for d in outDF['DateTime']]
outDF['microseconds'] = [d.time().microsecond for d in outDF['DateTime']]
# Create df to output
out_df = outDF[
    ['TIMESTAMP', 'RECORD', 'Counter', 'DepthCm', 'BattVolts', 'latitude_a', 'latitude_b', 'longitude_a', 'longitude_b',
     'Q', 'ns', 'height(m)', 'DepthVolts', 'latitudeDDDDD', 'longitudeDDDDD', 'month', 'dayofmonth', 'hourofday',
     'minutes', 'seconds', 'microseconds', 'latitude(deg)', 'longitude(deg)', 'sdn(m)', 'sde(m)', 'sdu(m)', 'sdne(m)']]
out_df = out_df.rename(columns={'Q': 'fix_quality', 'ns': 'nmbr_satellites', 'height(m)': 'altitudeB'})
import matplotlib.pyplot as plt
import seaborn as sns

# create boxplot with a different y scale for different rows
labels = ["Fix", "Float", "Single"]
groups = outDF.groupby('Q')
fig, axes = plt.subplots(1, len(labels))
i = 0
for ID, group in groups:
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
# print("Writing out: %s" % out_csv_fn)
# out_df.to_csv(out_csv_fn, index=False)
# out_df.to_feather(out_csv_fn.split(".")[0]+".feather")