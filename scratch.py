import pandas as pd
import geopandas as gpd
#import argparse
#import numpy as np
from shapely.geometry import Point
import matplotlib.pyplot as plt

raw_file = '/mnt/data/UAF-data/raw/SALVO/20240419/magnaprobe/ARM_magnaprobe_200m_GEODEL_20240419.a0.dat'


# Read MagnaProbe raw data (*.dat) into a DataFrame
header_row = 1
junk_header = [2, 3]
raw_df = pd.read_csv(raw_file, header=header_row)

# Drop junk su  header
raw_df = raw_df.drop(raw_df.index[:2])

# parse timesteamp
raw_df['datetime'] = pd.to_datetime(raw_df['TIMESTAMP'])

# coordinate columns:
raw_df['Latitude'] = raw_df['latitude_a'].astype('float') + raw_df['LatitudeDDDDD'].astype('float')
raw_df['Longitude'] = raw_df['Longitude_a'].astype('float') + raw_df['LongitudeDDDDD'].astype('float')

# depth in m not cm
raw_df['Snow Depth'] = raw_df['DepthCm'].astype('float') / 100


# Create geospatial references with GS84
# GeoDataFrame WGS84
raw_df['geometry'] = raw_df.apply(lambda x: Point((float(x['Longitude']), float(x['Latitude']))), axis=1)
gdf = gpd.GeoDataFrame(raw_df, geometry='geometry', crs='EPSG:4326')

# ESPG for Alaska 3338
gdf = gdf.to_crs('EPSG:3338')
raw_df['distance'] = gdf.distance(gdf.shift(1))
raw_df['linear distance'] = raw_df['distance'].cumsum()

# Data Cleaner
raw_df['quality'] = 0
raw_df.loc[raw_df['Snow Depth'] < 0, 'quality'] = 1

# check if max value before or after min/small value (0.5mm) or 119 +
cal_low = 0.02
cal_upp = 1.18

# consecutive distance is smaller than half distance between consecutive point
d_mean = raw_df['distance'].mean()
raw_df.loc[raw_df['distance'] < d_mean / 2, 'quality'] = 2

# consecutive time is smaller than 1 seconds
import datetime as dt
raw_df.loc[raw_df['datetime'].diff() < dt.timedelta(0, 1), 'quality'] = 3

# consecutive distance is smaller than half distance between point


#
# # Rolling mean
# raw_df['hs_mov_avg'] = raw_df['Snow Depth'].rolling(5).mean()
# raw_df['hs_mov_std'] = raw_df['Snow Depth'].rolling(5).std()
# raw_df.loc[(raw_df['Snow Depth'] < raw_df['hs_mov_avg'] - 3 * raw_df['hs_mov_std']) | \
#            (raw_df['hs_mov_avg'] + 3 * raw_df['hs_mov_std'] < raw_df['Snow Depth']), 'quality'] = 1


import matplotlib.pyplot as plt

plt.figure();
raw_df.plot(x='linear distance', y='Snow Depth')


# Plot snow depth as coordinate
plt.figure()
gdf.plot(column='Snow Depth')
plt.show()
