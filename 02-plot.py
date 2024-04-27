import pandas as pd
import geopandas as gpd
import os
#import argparse
import numpy as np
from shapely.geometry import Point
import matplotlib.pyplot as plt

ice1 = '/mnt/data/UAF-data/raw_0/SALVO/20240417/magnaprobe/ICE_magnaprobe_20140417.a2.csv'
ice2 = '/mnt/data/UAF-data/raw_0/SALVO/20240421/magnaprobe/ICE_GEODEL_200m_20240421.a3.csv'


# Read MagnaProbe raw data (*.dat) into a DataFrame
header_row = 0
date_cols = [1]
raw_1_df = pd.read_csv(ice1, header=header_row,parse_dates=date_cols)
raw_2_df = pd.read_csv(ice2, header=header_row,parse_dates=date_cols)

fig = plt.figure()
plt.plot(raw_1_df['linear distance'], raw_1_df['Snow Depth'], marker='o')
plt.plot(raw_2_df['linear distance'], raw_2_df['Snow Depth'], marker='x')
plt.show()
