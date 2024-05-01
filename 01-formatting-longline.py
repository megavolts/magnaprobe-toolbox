#!/usr/bin/python3
# -*- coding: utf-8 -*-

from cmcrameri import cm
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import gridspec as gridspec
import logging
import numpy as np
import os
import pandas as pd
import scipy.stats as stats

#import pint
#import pint_pandas as pint

logger = logging.getLogger(__name__)

# -- USER VARIABLES
# Enable plotting with relative coordinate to the transect origin x0=0, y0=0
plot_origin_flag = True

# Data filename
data_fp = '/mnt/data/UAF-data/working_a/SALVO/20240416-ARM/magnaprobe/salvo_arm_longline_magnaprobe-geo1_20240416.a2.csv'


def output_filename(in_fp):
    """
    Generate output filepath base on the input raw filepath. If the file directory does not exist it is created.
    :param in_fp: str
    :return: str
        A string containing the filename in which the output data is writtent
    """
    base_dir = os.path.dirname(in_fp)
    if '/raw/' in base_dir:
        out_dir = base_dir.replace('/raw/', '/working_a/')
    else:
        out_dir = base_dir

    # Input filename
    input_fn = os.path.basename(in_fp)
    if '.00.' in input_fn:
        output_fn = input_fn.replace('.00.', '.a1.')
    elif '.a' in input_fn:
        _increment_number = int(input_fn.split('.')[-2][1:])
        _next_number = _increment_number + 1
        if _next_number == 0:
            # According to ARM guideline a0 is only for raw data exported to NetCDF
            # https://www.arm.gov/guidance/datause/formatting-and-file-naming-protocols
            _next_number = 1
        output_fn = input_fn.replace(str('a%.0f' % _increment_number), str('a%.0f' % _next_number))
    out_file = os.path.join(out_dir, output_fn)
    out_dir = ('/').join(out_file.split('/')[:-1])
    if not os.path.exists(out_dir):
        logger.info(str('Creating output file directory: %s' %out_dir))
        os.makedirs(('/').join(out_file.split('/')[:-1]))
    return out_file


def import_qc_data(df):
    """
    Read raw data file from filepath raw_fp.
    :param df: string
        Filepath to the magnaprobe raw data file.
    :param header_row: int
    :return:
    """
    if not df.endswith('.csv'):
        logger.warning('This may not be a QC-ed datafile, extension does not match.')
    else:
        pass

    df = pd.read_csv(df, header=[0])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df


def euclidian_distance(x1, x2, y1, y2, z1=0, z2=0):
    """
    Calculates euclidian distance between 2 points P(x1, y1, z1) and P(x2, y2, z2)
    If no elevation is given, it is set to 0 m.
    x1, x2, y1, y2 must have the same dimension; when given z1 and z2 must either be the same dimension as x1, or be a
    float that will be propagated into an array of a same dimension as x1

    :param x1 : array_like, float
        distance along x from origin x=0 for point P1
    :param y1 : array_like, float
        distance along y from origin y=0 for point P1
    :param x2 : array_like, float
        distance along x from origin x=0 for point P2
    :param y2 : array_like, float
        distance along y from origin y=0 for point P2
    :param z1 : optional, array_like, float
        distance along z from origin z=0 for point P1
    :param z2 : optional, array_like, float
        distance along z from origin z=0 for point P2

    :return d: ndarray, float
        The calculated euclidian distance in meter [m]
    """
    #TODO: use Pi(X1, Y1, Z1) rather than single coordinate
    if isinstance(x1, (int, float, list)):
        x1 = pd.Series([x1]*len(x2), dtype=float)
    #     x1 = pd.Series([float(x1)]*len(x2), dtype='pint[m]')
    # elif isinstance(x1, pint.Quantity):
    #     x1 = pd.Series([x1.to_base_units().magnitude.astype(float)]*len(x2), dtype='pint[m]')
    if isinstance(x2, (int, float, list)):
        x2 = pd.Series([x2] * len(x1), dtype=float)
    #     x2 = pd.Series([float(x2)]*len(x1), dtype='pint[m]')
    # elif isinstance(x2, pint.Quantity):
    #     x2 = pd.Series([x2.to_base_units().magnitude.astype(float)]*len(x1), dtype='pint[m]')
    if isinstance(y1, (int, float, list)):
        y1 = pd.Series([z1]*len(x1), dtype=float)
    #     y1 = pd.Series([float(y1)]*len(x1), dtype='pint[m]')
    # elif isinstance(y1, pint.Quantity):
    #     y1 = pd.Series([y1.to_base_units().magnitude.astype(float)]*len(x1), dtype='pint[m]')
    if isinstance(y2, (int, float, list)):
        y2 = pd.Series([y2] * len(x1), dtype=float)
    #     y2 = pd.Series([float(y2)]*len(x1), dtype='pint[m]')
    # elif isinstance(y2, pint.Quantity):
    #     y2 = pd.Series([y2.to_base_units().magnitude.astype(float)]*len(x1), dtype='pint[m]')

    # convert float into pd.Series with pint
    if isinstance(z1, (int, float, list)):
        z1 = pd.Series([z2] * len(x1), dtype=float)
    #     z1 = pd.Series([float(z1)]*len(x1), dtype='pint[m]')
    # elif isinstance(z1, pint.Quantity):
    #     z1 = pd.Series([z1.to_base_units().magnitude.astype(float)]*len(x1), dtype='pint[m]')
    if isinstance(z2, (int, float, list)):
        z2 = pd.Series([z2] * len(x1), dtype=float)
    #     z2 = pd.Series([float(z2)]*len(x1), dtype='pint[m]')
    # elif isinstance(z2, pint.Quantity):
    #     z2 = pd.Series([z2.to_base_units().magnitude.astype(float)]*len(x1), dtype='pint[m]')
    if x1.shape != x2.shape \
            or x1.shape != y1.shape or x1.shape != y2.shape \
            or x1.shape != z1.shape or x1.shape != z2.shape:
        logger.error('x1, x2, y1, y2, z1 and z2 must all have the same dimensions')
        return 0

    d = ((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**0.5
    return d


def compute_distance(df, origin_pt=None):
    """
    Compute distance in between point (TrackDist), cumulative distance in between point (TrackDistCum) and distance from
    origin point (DistOrigin). The origin point is the first point of the dataframe, unless
    :param df: pd.DataFrame()
            Dataframe containing the raw data. Datsaframe should contain columns X, Y and Z. The latter could be set
            to 0 if not needed.
    :param origin_pt: 1darray or None (default)
        If none, origin point is the first point in the dataframe
        Origin point should be either of format [x0, y0, z0] or [x0, y0]. In the latter case, z0 is set to zero (z0=0).
    :return:
    """
    if origin_pt is None:
        x0 = df.iloc[0]['X']
        y0 = df.iloc[0]['Y']
        z0 = df.iloc[0]['Z']
    elif isinstance(origin_pt, (list)) and len(list) <= 3:
        x0 = origin_pt[0]
        y0 = origin_pt[1]
        try:
            z0 = origin_pt[2]
        except IndexError:
            logger.warning('Z-coordinate of origin point, z0, is not defined. Setting z0=1')
            z0 = 0
        else:
            pass
    else:
        logger.error('TODO: compute_distance not defined')

    df['TrackDist'] = euclidian_distance(df['X'], df['X'].shift(), df['Y'], df['Y'].shift())
    df.iloc[0, df.columns.get_loc('TrackDist')] = 0
    # Compute cumulative sum of distance between two consecutive points
    df['TrackDistCum'] = df['TrackDist'].cumsum()
    df.iloc[0, df.columns.get_loc('TrackDistCum')] = 0
    # Compute distance between the origin point and the current point
    df['DistOrigin'] = euclidian_distance(df['X'], x0, df['Y'], y0, df['Z'], z0)

    return df


def calibration_check(df, lower_cal=0.02, upper_cal=1.19):
    """
    Check magnaprobe data for calibration point.

    - Quality flag None: no quality check performed
    - Quality flag 0: quality check performed to mark

    Quality flag
    - Quality flag 1: good point
    - Quality flag 4: bad point

    calibration flag
    - Quality flag None: not a calibration point
    - Quality flag 'U': upper calibration point possible
    - Qulaity flag 'L': lower calibration point possible

    :param df: pd.DataFrame()
        Dataframe containing the raw data, with snow depth in m in column `Snowdepth`
    :param lower_cal: float, default 0.02 m
        Lower snow depth under which measurement could have been a lower calibration point
    :param upper_cal: float default 1.18 m
        Upper snow depth above which measurement could have been an upper calibration point
    :return: pd.DataFrame()
        Dataframe containing the raw data, augmented with a column named `Calibration` containing the location of
        possible calibration point

    """
    # Set Calibration column to None
    df['Calibration'] = None
    # TODO: to be implemented
    # Look for lower calibration point
    df.loc[df['SnowDepth'] < lower_cal, 'Calibration'] = 'L'
    # Look for upper calibration point
    df.loc[upper_cal < df['SnowDepth'], 'Calibration'] = 'U'

    # Look for typical calibration pattern 'UL', 'LU', 'ULU', 'LUL', ...

    return df


def quality_check(df):
    """
    Check quality of the magnaprobe data. Data quality is stored in `quality` column with the following flag:
    - Quality flag None: no quality check performed
    - Quality flag 0: quality check performed to mark

    Quality flag
    - Quality flag 1: good point
    - Quality flag 4: bad point

    Error flag
    - Quality flag 7: snow depth is negative
    - Quality flag 8: distance between consecutive points is too small (less than 1/3 of median TrackDist distance)
    - Quality flag 9: time between consecutive points is too short (less than 1 second)
    :param df: pd.DataFrame()
        Dataframe containing the raw data, with snow depth in m in column `Snowdepth`
    :return: pd.DataFrame()
        Dataframe containing the raw data, augmented with a columns titled `Quality` containing the quality flag

    """
    # Set all Quality flag to 0
    df['Quality'] = 0

    # Negative snow depths
    df.loc[df['SnowDepth'] < 0, 'Quality'] = 7

    # Too short spatial difference
    d_median = df['TrackDist'].median()
    df.loc[df['TrackDist'] < d_median / 3, 'Quality'] = 8

    # Too short time difference
    df.loc[df['Timestamp'].diff() < dt.timedelta(0, 0.9), 'Quality'] = 9

    print(df.loc[df['Quality'] > 6, ['Record', 'Counter', 'Timestamp', 'SnowDepth', 'Quality']])

    return df


def stat_basics(y_data):
    """
    Compute basic statistic (sum, average, median, maximum, minimum, minimum location, maximum location
    standard deviation, root mean squared, skewness, kurtosis)
    :param Y: array, pd.Series(), pd.DataFrame()
    :return: dict
        Dictionnary containing basic statistic for the array
    """
    hs_stats = {}
    hs_stats['sum'] = np.sum(y_data)
    hs_stats['avg'] = stats.tmean(y_data)
    hs_stats['med'] = np.median(y_data)
    hs_stats['max'] = stats.tmax(y_data)
    hs_stats['maxloc'] = np.where(y_data == hs_stats['max'])[0]
    hs_stats['min'] = stats.tmin(y_data)
    hs_stats['minloc'] = np.where(y_data == hs_stats['min'])[0]
    hs_stats['sdev'] = stats.tstd(y_data)  # Standard deviation
    hs_stats['adev'] = np.sum(np.abs((y_data - np.mean(y_data))))/len(y_data)
    hs_stats['sem'] = stats.tsem(y_data)  # Standard error of mean
    hs_stats['rms'] = np.sqrt(np.sum(y_data**2)/len(y_data))  # Root mean squared
    hs_stats['skew'] = stats.skew(y_data, bias=False)
    hs_stats['kurt'] = stats.kurtosis(y_data, bias=False)
    return hs_stats


def export_data(df, out_fp, header_order, drop_header=True):
    """
    :param raw_df: pd.DataFrame()
        Dataframe containing the data to export
    :param out_fp: string
        Output filename.
    :param header_order: list of string
        List containing the ordered headers
    :param drop_header: boolean
        If True, drop any headers not in the list.
        If False, append existing headers not in the ordered header list after the header list.
    :return:
    """
    df.columns = [c[0].upper() + c[1:] for c in df.columns]
    if not drop_header:
        header_order = header_order + [col for col in df.columns if col not in header_order]
    else:
        pass
    raw_df = df[header_order]
    raw_df.to_csv(out_fp, index=False)
    print(out_fp)

out_fp = output_filename(data_fp)
data_df = import_qc_data(data_fp)

# Compute the distance between points to take in account any removed rows.
data_df = compute_distance(data_df)
data_df = quality_check(data_df)
data_df = calibration_check(data_df)

if len(data_df[data_df.Quality > 6]) > 0:
    print('Cleaning is still needed')

# # export QC, cleaned data file
header_order = ['Record', 'Counter', 'Timestamp', 'SnowDepth', 'Latitude', 'Longitude', 'X', 'Y', 'Z', 'TrackDist',
                'TrackDistCum', 'DistOrigin', 'Quality', 'Calibration', 'Battvolts']
export_data(data_df, out_fp, header_order)

def compute_histogram(y_data, bins=None):
    """
    Compute the snow histogram
    :param y_data:
    :return:
    """
    if bins is None:
        # round maximum snow depth to the nearest decimeters
        y_max = np.ceil(y_data.max()*10)/10
        # minimum snow depth is 0
        y_min = 0
        # snow depth increment is 0.1 m
        d_y = 0.1
        bins = np.arange(y_min, y_max+d_y/2, d_y)
    else:
        pass
    counts, bins = np.histogram(y_data, bins, density=True)
    hy_bin_mid = bins[:-1] + np.diff(bins) / 2
    df = pd.DataFrame(data=counts, index=hy_bin_mid, columns=['Count'])
    df.index.name = 'y_mid'
    return df


def compute_pdf(y_data):
    pdf_p = stats.lognorm.fit(y_data)
    pdf_f = stats.lognorm(*pdf_p)
    return pdf_f
# Basic stats
hs_stat = stat_basics(data_df['SnowDepth'])

# Histogram and pdf
y_data = data_df['SnowDepth']
hs_histogram = compute_histogram(y_data)
hs_pdf_fit = compute_pdf(data_df['SnowDepth'])
hs_x = np.arange(0, y_data.max(), 0.01)

# Plot basic figures:
site = out_fp.split('_')[2].upper()
date = out_fp.split('.')[0][-8:]
if '_longline' in out_fp:
    name = 'Long transect'
elif '_line' in out_fp:
    name = '200 m transect'
elif 'library' in out_fp:
    name = 'Library site'

# Data status plot
plot_df = data_df
# Move origin points to x0, y0
if plot_origin_flag:
    plot_df['x0'] = plot_df['X'] - plot_df.iloc[0]['X']
    plot_df['y0'] = plot_df['Y'] - plot_df.iloc[0]['Y']
else:
    plot_df['x0'] = plot_df['X']
    plot_df['y0'] = plot_df['Y']
snow_depth_scale = plot_df['SnowDepth'].max()-plot_df['SnowDepth'].min()
z0 = cm.davos(plot_df['SnowDepth']/snow_depth_scale)
plt.style.use('ggplot')

# Define figure subplots
nrows, ncols = 4, 1
w_fig, h_fig = 8, 11
fig = plt.figure(figsize=[w_fig, h_fig])
gs1 = gridspec.GridSpec(nrows, ncols, height_ratios=[1, 1, 1, 1], width_ratios=[1])
ax = [[fig.add_subplot(gs1[0, 0])], [fig.add_subplot(gs1[1, 0])], [fig.add_subplot(gs1[2, 0])], [fig.add_subplot(gs1[3, 0])]]
ax = np.array(ax)
ax[0, 0].plot(plot_df['TrackDistCum'], plot_df['SnowDepth'], color='steelblue')
ax[0, 0].fill_between(plot_df['TrackDistCum'], plot_df['SnowDepth'], [0]*len(plot_df), color='lightsteelblue')

ax[0, 0].set_xlabel('Distance along the transect (m)')
ax[0, 0].set_ylabel('Snow Depth (m)')
ax[0, 0].set_xlim([0, plot_df['TrackDistCum'].max()])
ax[0, 0].set_ylim([0, 1.2])

ax[1, 0].scatter(plot_df['x0'], plot_df['y0'], c=z0)
if plot_origin_flag:
    ax[1, 0].set_xlabel('X coordinate (m)')
    ax[1, 0].set_ylabel('Y coordinate (m)')
else:
    ax[1, 0].set_xlabel('UTM X coordinate (m)')
    ax[1, 0].set_ylabel('UTM Y coordinate (m)')

# Snow depth probability function:
ax[2, 0].bar(hs_histogram.index, hs_histogram['Count'], 0.1, edgecolor='steelblue', color='lightsteelblue')
ax[2, 0].plot(hs_x, hs_pdf_fit.pdf(hs_x), 'k')
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
