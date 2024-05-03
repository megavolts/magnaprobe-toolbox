
__name__ = 'magnaprobe_toolbox.io.tools'
__all__ = ['strip_columns', 'quality_check']

import datetime as dt
import logging
from magnaprobe_toolbox.io import col2remove_l
from magnaprobe_toolbox.io import lower_cal, upper_cal
import pandas as pd
logger = logging.getLogger(__name__)


def strip_columns(df, col2remove_l=col2remove_l):
    """
    Remove unused columns, according to the list of columns to strip header2strip
    :param df: pd.DataFrame()
            Dataframe containing the raw data.
    :param col2remove_l:
    :return:
    """
    col2remove_l = [h[0].upper()+h[1:].lower() for h in col2remove_l]
    df.drop(columns=col2remove_l, inplace=True)
    return df


def quality_check(raw_df, display=False):
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
    :param display: boolean
        If True, display flag rows of dataframe
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

    if display:
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


def all_check(raw_df, display=False, lower_cal=lower_cal, upper_cal=upper_cal):
    raw_df = quality_check(raw_df, display=display)
    raw_df = calibration_check(raw_df, lower_cal=lower_cal, upper_cal=upper_cal)
    return raw_df


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

    # TODO: if previous point is missing, compute with the last known coordiante, aka filter nas row out
    # USE 2040417-ICE
    raw_df['TrackDist'] = euclidian_distance(raw_df['X'], raw_df['X'].shift(), raw_df['Y'], raw_df['Y'].shift())
    raw_df.iloc[0, raw_df.columns.get_loc('TrackDist')] = 0
    # Compute cumulative sum of distance between two consecutive points
    raw_df['TrackDistCum'] = raw_df['TrackDist'].cumsum()
    raw_df.iloc[0, raw_df.columns.get_loc('TrackDistCum')] = 0
    # Compute distance between the origin point and the current point
    raw_df['DistOrigin'] = euclidian_distance(raw_df['X'], x0, raw_df['Y'], y0)

    return raw_df
# Quality check
