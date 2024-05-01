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
#import pint
#import pint_pandas as pint
import pyproj

logger = logging.getLogger(__name__)

# -- USER VARIABLES
# Enable plotting with relative coordinate to the transect origin x0=0, y0=0
plot_origin_flag = True
# ESPG code for local projection
local_ESPG = '3338'  # for Alaska
# Magnaprobe calibration value
lower_cal = 0.02  # m
upper_cal = 1.18  # m

# Define
# ureg = pint.UnitRegistry(auto_reduce_dimensions=True)
# pint.set_application_registry(ureg)
#ureg = pint.get_application_registry()
# lower_cal = lower_cal * ureg.m # m
# upper_cal = upper_cal * ureg.m # m

# List of headers to strip
col2remove_l = ['latitude_a', 'latitude_b', 'Longitude_a', 'Longitude_b', 'fix_quality',
               'nmbr_satellites', 'HDOP', 'altitudeB', 'DepthVolts', 'LatitudeDDDDD', 'LongitudeDDDDD', 'month',
               'dayofmonth', 'hourofday', 'minutes', 'seconds', 'microseconds', 'depthcm']

# Filename
raw_fp = '/mnt/data/UAF-data/raw/SALVO/20240416-ARM/magnaprobe/salvo_arm_longline_magnaprobe-geo1_20240416.00.dat'

# Define output filename
def output_filename(in_fp):
    """
    Generate output filepath base on the input raw filepath. If the file directory does not exist it is created.
    :param in_fp: str
    :return: str
        A string containing the filename in which the output data is writtent
    """
    base_dir = os.path.dirname(in_fp)
    out_dir = base_dir.replace('/raw/', '/working_a/')

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


# Read MagnaProbe raw data (*.dat) into a DataFrame
def import_raw(raw_fp, header_row=1):
    """
    Read raw data file from filepath raw_fp.
    :param raw_fp: string
        Filepath to the magnaprobe raw data file.
    :param header_row: int
    :return:
    """
    logger = logging.getLogger(__name__)
    if not raw_fp.endswith('.dat'):
        logger.warning('This may not be an original raw datafile, extension is not ".dat"')
    else:
        pass

    raw_df = pd.read_csv(raw_fp, header=header_row)
    return raw_df

def format_col_headers(raw_fp):
    """
    Format all columns header in lower case, but the first letter.
    :param raw_fp: pd.DataFrame()
        Dataframe containing the raw data, imported with read_row.
    :return:
        Dataframe with formatted column headers
    """
    # set all columns headers in lower case
    raw_df.columns = [c[0].upper()+c[1:].lower() for c in raw_df.columns]
    return raw_df

def remove_junk_row(raw_df):
    """
    Remove junk rows, line 3 and 4 in the original magnaprobe raw datafile
    :param raw_df: pd.DataFrame()
        Dataframe containing the raw data.
    :return: pd.DataFrame()
        Dataframe stripped from the 2 first junk rows.
    """

    raw_df = raw_df.drop(raw_df.index[:2])
    return raw_df


def check_datetime(raw_df, date=dt.date(2020,1,1)):
    """
    Format timestamp to ISO. If timestamps are invalid, artificial timestamps are generated starting at midnight local
    time, with 1-second increments.

    :param raw_df: pd.DataFrame()
        Dataframe containing the raw data.
    :param date: dt.datetime()
        Date at which the dataset was acquired
    :return:
        Dataframe with formatted timestamp.
    """
    logger = logging.getLogger(__name__)

    try:
        raw_df['Timestamp'] = pd.to_datetime(raw_df['Timestamp'], format='ISO8601')
    except ValueError:
        # Generate artificial timestamp starting at midnight local time
        date = pd.to_datetime(date)
        raw_df['Timestamp'] = pd.date_range(date, periods=len(raw_df), freq="s")
        logger.info(str("Time data is not ISO8601 format. Artificial timestamp is generated starting at midnight local"
                        "time of %s" % date.strftime("%Y%m%d")))
    else:
        pass
    return raw_df

def compute_coordinate(raw_df):
    """
    Compute latitude and longitude coordinate in degree from the integer (_a) and minute (_b) fields.

    :param raw_df: pd.DataFrame()
        Dataframe containing the raw data.
    :return: pd.DataFrame()
        Dataframe containing the latitude, respectively longitude, in the Latitude, respectively Longitude columns,
        given in Decimal Degree.
    """
    raw_df['Latitude'] = raw_df['Latitude_a'].astype('float') + raw_df['Latitude_b'].astype('float')/60
    raw_df['Longitude'] = raw_df['Longitude_a'].astype('float') + raw_df['Longitude_b'].astype('float')/60

    # # set units
    # raw_df['Latitude'] = raw_df['Latitude'].astype('pint[degree]')
    # raw_df['Longitude'] = raw_df['Longitude'].astype('pint[degree]')
    return raw_df

def convert_snowdepth_to_m(raw_df):
    """
    Convert snow depth from cm to m.

    :param raw_df: pd.DataFrame()
        Dataframe containing the raw data.
    :return: pd.DataFrame()
        Dataframe containing the snow depth given in meter (m).
    """
    raw_df['SnowDepth'] = raw_df['Depthcm'].astype(float) / 100.0
    raw_df['SnowTension'] = raw_df['Depthvolts'].astype(float)

    # # set units
    # raw_df['SnowDepth'] = raw_df['SnowDepth'].astype('pint[m]')
    # raw_df['SnowTension'] = raw_df['SnowTension'].astype('pint[V]')
    return raw_df

def convert_wgs_to_utm(raw_df, local_ESPG=local_ESPG):
    """
    Convert latitude/longitude degree coordinates from WSG84 into local projection (AK ESPGL 3338).
    By default, altitude Z=0.

    :param raw_df: pd.DataFrame()
        Dataframe containing the raw data.
    :param local_ESPG: int, default local
    :return: pd.DataFrame()
        Dataframe augmented with the columns X, Y and Z containing the latitude/longitude coordinate projected locally
        in m.
    """
    # TODO: include elevation Z
    # Create X, Y geospatial in ESPG 3338 reference for Alaska
    xform = pyproj.Transformer.from_crs('4326', local_ESPG)
    raw_df['X'], raw_df['Y'] = xform.transform(raw_df['Latitude'], raw_df['Longitude'])
    raw_df['Z'] = [0.]*len(raw_df)

    # # set units
    # raw_df['X'] = raw_df['X'].astype('pint[m]')
    # raw_df['Y'] = raw_df['Y'].astype('pint[m]')
    # raw_df['Z'] = raw_df['Z'].astype('pint[m]')
    return raw_df

def strip_columns(raw_df, col2remove_l=col2remove_l):
    """
    Remove unused columns, according to the list of columns to strip header2strip
    :param raw_df: pd.DataFrame()
            Dataframe containing the raw data.
    :param col2remove_l:
    :return:
    """
    col2remove_l = [h[0].upper()+h[1:].lower() for h in col2remove_l]
    raw_df.drop(columns=col2remove_l, inplace=True)
    return raw_df


out_fp = output_filename(raw_fp)

raw_df = import_raw(raw_fp)
raw_df = format_col_headers(raw_fp)
raw_df = remove_junk_row(raw_df)
raw_df = check_datetime(raw_df, out_fp.split('.')[0][-8:])
raw_df = compute_coordinate(raw_df)
raw_df = convert_snowdepth_to_m(raw_df)
raw_df = convert_wgs_to_utm(raw_df)
raw_df = strip_columns(raw_df)
raw_df.dtypes

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

    # check if x1, x2, y1, y2, z1 and z2
    d = ((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**0.5
    return d
def compute_distance(raw_df, origin_pt=None):
    """
    Compute distance in between point (TrackDist), cumulative distance in between point (TrackDistCum) and distance from
    origin point (DistOrigin). The origin point is the first point of the dataframe, unless
    :param raw_df: pd.DataFrame()
            Dataframe containing the raw data. Datsaframe should contain columns X, Y and Z. The latter could be set
            to 0 if not needed.
    :param origin_pt: 1darray or None (default)
        If none, origin point is the first point in the dataframe
        Origin point should be either of format [x0, y0, z0] or [x0, y0]. In the latter case, z0 is set to zero (z0=0).
    :return:
    """
    if origin_pt is None:
        x0 = raw_df.iloc[0]['X']
        y0 = raw_df.iloc[0]['Y']
        z0 = raw_df.iloc[0]['Z']
    elif isinstance(origin_pt, (list)) and len(list) <= 3:
        x0 = origin_pt[0]
        y1 = origin_pt[1]
        try:
            z0 = origin_pt[2]
        except IndexError:
            logger.warning('Z-coordinate of origin point, z0, is not defined. Setting z0=1')
            z0 = 0
        else:
            pass
    else:
        logger.error('TODO: compute_distance not defined')

    raw_df['TrackDist'] = euclidian_distance(raw_df['X'], raw_df['X'].shift(), raw_df['Y'], raw_df['Y'].shift())
    raw_df.iloc[0, raw_df.columns.get_loc('TrackDist')] = 0
    # Compute cumulative sum of distance between two consecutive points
    raw_df['TrackDistCum'] = raw_df['TrackDist'].cumsum()
    raw_df.iloc[0, raw_df.columns.get_loc('TrackDistCum')] = 0
    # Compute distance between the origin point and the current point
    raw_df['DistOrigin'] = euclidian_distance(raw_df['X'], x0, raw_df['Y'], y0)

    return raw_df
# Quality check
def quality_check(raw_df):
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
    :param raw_df: pd.DataFrame()
        Dataframe containing the raw data, with snow depth in m in column `Snowdepth`
    :return: pd.DataFrame()
        Dataframe containing the raw data, augmented with a columns titled `Quality` containing the quality flag

    """
    # Set all Quality flag to 0
    raw_df['Quality'] = 0

    # Negative snow depths
    raw_df.loc[raw_df['SnowDepth'] < 0, 'Quality'] = 7

    # Too short spatial difference
    d_median = raw_df['TrackDist'].median()
    raw_df.loc[raw_df['TrackDist'] < d_median / 3, 'Quality'] = 8

    # Too short time difference
    raw_df.loc[raw_df['Timestamp'].diff() < dt.timedelta(0, 0.9), 'Quality'] = 9

    print(raw_df.loc[raw_df['Quality'] > 6, ['Record', 'Counter', 'Timestamp', 'SnowDepth', 'Quality']])

    return raw_df

def calibration_check(raw_df, lower_cal=lower_cal, upper_cal=upper_cal):
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

    :param raw_df: pd.DataFrame()
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
    raw_df['Calibration'] = None
    # TODO: to be implemented
    # Look for lower calibration point
    raw_df.loc[raw_df['SnowDepth'] < lower_cal, 'Calibration'] = 'L'
    # Look for upper calibration point
    raw_df.loc[upper_cal < raw_df['SnowDepth'], 'Calibration'] = 'U'

    # Look for typical calibration pattern 'UL', 'LU', 'ULU', 'LUL', ...

    return raw_df

def export_data(raw_df, out_fp, header_order, drop_header=True):
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
    raw_df.columns = [c[0].upper() + c[1:] for c in raw_df.columns]
    if not drop_header:
        header_order = header_order + [col for col in raw_df.columns if col not in header_order]
    else:
        pass
    raw_df = raw_df[header_order]
    raw_df.to_csv(out_fp, index=False)
    print(out_fp)

raw_df = compute_distance(raw_df)
raw_df = quality_check(raw_df)
raw_df = calibration_check(raw_df)

header_order = ['Record', 'Counter', 'Timestamp', 'SnowDepth', 'Latitude', 'Longitude', 'X', 'Y', 'Z', 'TrackDist',
                'TrackDistCum', 'DistOrigin', 'Quality', 'Calibration', 'Battvolts']
export_data(raw_df, out_fp, header_order)

### PLOT:
#
#
#
#
#
#
# # Data status plot
# # Move origin points to x0, y0
# if plot_origin_flag:
#     raw_df['x0'] = raw_df['X'] - raw_df.iloc[0]['X']
#     raw_df['y0'] = raw_df['Y'] - raw_df.iloc[0]['Y']
# else:
#     raw_df['x0'] = raw_df['X']
#     raw_df['y0'] = raw_df['Y']
#
# snowdepth_delta = raw_df['SnowDepth'].max() - raw_df['SnowDepth'].min()
# #raw_df.loc[raw_df['SnowDepth'] < 0*ureg.m, 'SnowDepth'] = 0
# # TODO: find a better way
# #hs = [raw_df['SnowDepth'].values[ii].magnitude for ii in range(0, raw_df.__len__())]
# #z0 = cm.davos(hs/snowdepth_delta.magnitude)
# z0 = cm.davos(raw_df['SnowDepth']/snowdepth_delta)
#
# # Define figure subplots
# nrows, ncols = 2, 1
# w_fig, h_fig = 8, 11
# fig = plt.figure(figsize=[w_fig, h_fig])
# gs1 = gridspec.GridSpec(nrows, ncols, height_ratios=[1, 1], width_ratios=[1])
# ax = [[fig.add_subplot(gs1[0, 0])], [fig.add_subplot(gs1[1, 0])]]
# ax = np.array(ax)
# ax[0, 0].plot(raw_df['TrackDistCum'].values, raw_df['SnowDepth'].values, color='steelblue')
# ax[0, 0].fill_between(raw_df['TrackDistCum'], raw_df['SnowDepth'], [0]*len(raw_df), color='lightsteelblue')
#
# ax[0, 0].set_xlabel('Distance along the transect (m)')
# ax[0, 0].set_ylabel('Snow Depth (m)')
# ax[0, 0].set_xlim([0, raw_df['TrackDistCum'].max()])
# ax[0, 0].set_ylim([0, 1.2])
#
# ax[1, 0].scatter(raw_df['x0'], raw_df['y0'], c=z0)
# ax[1, 0].set_xlabel('X coordinate (m)')
# ax[1, 0].set_ylabel('Y coordinate (m)')
#
# if np.diff(ax[1, 0].get_ylim()) < 200:
#     y_min = min(ax[1, 0].get_ylim())
#     y_max = max(ax[1, 0].get_ylim())
#     y_avg = np.mean([y_min, y_max])
#     ax[1, 0].set_ylim([y_avg-100, y_avg+100])
# plt.show()
