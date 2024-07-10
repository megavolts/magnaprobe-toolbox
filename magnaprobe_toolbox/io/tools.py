__name__ = 'magnaprobe_toolbox.io.tools'
__all__ = ['strip_columns', 'quality_check']
__author__ = 'Marc Oggier'
from datetime import timedelta
import logging
from magnaprobe_toolbox.io import col2remove_l
from magnaprobe_toolbox.io import lower_cal, upper_cal
from magnaprobe_toolbox.analysis import distance
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
    - Quality flag 8: Distance between consecutive points is either small (less than 1/3 of the median distance) and
    points may be duplicate or large (more than 3 times the median distance and points may not be part of the transect.
    - Quality flag 9: time between consecutive points is too short (less than 1 second)
    :param raw_df: pd.DataFrame()
        Dataframe containing the raw data, with snow depth in m in column `Snowdepth`
    :param display: boolean
        If True, display flag rows of dataframe
    :return: pd.DataFrame()
        Dataframe containing the raw data, augmented with a columns titled `QC_flag` containing the quality flag
    """
    # Set all Quality flag to 0
    raw_df['QC_flag'] = 0

    # Negative snow depths
    raw_df.loc[raw_df['SnowDepth'] < 0, 'QC_flag'] = 7

    # Too short spatial difference
    d_median = raw_df['TrackDist'].median()
    raw_df.loc[raw_df['TrackDist'] < d_median / 3, 'QC_flag'] = 8
    raw_df.loc[d_median * 3 < raw_df['TrackDist'], 'QC_flag'] = 8

    # Too short time difference
    raw_df.loc[raw_df['Timestamp'].diff() < timedelta(0, 1), 'QC_flag'] = 9

    # At least twice above median sampling frequency
    frequency = raw_df.Timestamp.diff().median()
    # Twice above median sampling frequency
    raw_df.loc[frequency * 3 < raw_df.Timestamp.diff(), 'QC_flag'] = 9

    # At least twice below median sampling frequency
    raw_df.loc[raw_df.Timestamp.diff() < frequency /2, 'QC_flag'] = 8

    if display:
        display_quality_check(raw_df, [6, 7, 8])

    return raw_df

def display_quality_check(df, flag=None):
    """
    Display rows of corresponding quality flag, or where quality flag are bad (>3)
    :param df:
        Input DataFrame
    :param flag: array of integer
        Array of quality flag to display:
    :return None:
    """
    if flag is None:
        flag = [4, 5, 6, 7, 8, 9]
    print(df.loc[df['QC_flag'].isin(flag), ['Record', 'Counter', 'Timestamp',
                                            'SnowDepth', 'TrackDist', 'QC_flag']])
    return None


def calibration_check(raw_df, lower_cal=lower_cal, upper_cal=upper_cal):
    """
    Check magnaprobe data for calibration point.
    - Quality flag None: no quality check performed
    - Quality flag 0: quality check performed to mark

    Quality flag
    - Quality flag 1: good point
    - Quality flag 4: bad point

    Calibration flag
    - Quality flag None: not a calibration point
    - Quality flag 'U': upper calibration point possible
    - Qulaity flag 'L': lower calibration point possible

    :param raw_df: pd.DataFrame()
        Dataframe containing the raw data, with snow depth in m in column `SnowDepth`
    :param lower_cal: float, default 0.02 m
        Lower snow depth under which measurement could have been a lower calibration point
    :param upper_cal: float default 1.18 m
        Upper snow depth above which measurement could have been an upper calibration point
    :return: pd.DataFrame()
        Dataframe containing the raw data, augmented with a column named `Calibration` containing the location of
        possible calibration point

    """
    # Set Calibration column to None
    raw_df['CalPoint'] = None
    # TODO: to be implemented
    # Look for lower calibration point
    raw_df.loc[raw_df['SnowDepth'] < lower_cal, 'CalPoint'] = 'L'
    # Look for upper calibration point
    raw_df.loc[upper_cal < raw_df['SnowDepth'], 'CalPoint'] = 'U'

    # Look for typical calibration pattern 'UL', 'LU', 'ULU', 'LUL', ...

    return raw_df

def reorder(input_df, distance_type=None):
    if distance_type is not None:
        index_org_order = input_df.index
        input_df.sort_values(by=[distance_type], inplace=True)

        if not all(input_df.index == index_org_order) :
            logger.warning('Transect point has been reordered according to ' + distance_type)
    return input_df

def direction_check(df, direction='EW'):
    """
    Check if the direction of the transect measurement match the supposed direction of the transect. Return reverse
    dataframe if needed. Note that Decimal Degree should be given as absolute with 'W' and 'S' being negative.

    :param df: pd.DataFrame()
        Input DataFrame
    :param direction: 'EW', 'WE', 'NS', 'SN'
        Direction of the transect.
    :return df: pd.DataFrame()
        Output Dataframe with the row reorder in direction of the transect
    """
    reverse_direction_flag = False
    df.sort_values(by=['DistOrigin'])
    if direction == 'EW' and df.iloc[0]['Longitude'] < df.iloc[-1]['Longitude']:
        df = df.iloc[::-1]
        reverse_direction_flag = True
    elif direction == 'WE' and df.iloc[0]['Longitude'] > df.iloc[-1]['Longitude']:
        df = df.iloc[::-1]
        reverse_direction_flag = True

    if direction == 'NS' and df.iloc[0]['Latitude'] < df.iloc[-1]['Latitude']:
        df = df.iloc[::-1]
        reverse_direction_flag = True
    elif direction == 'SN' and df.iloc[0]['Latitude'] > df.iloc[-1]['Latitude']:
        df = df.iloc[::-1]
        reverse_direction_flag = True

    df = distance.compute(df)

    if reverse_direction_flag:
        logger.warning('Transect has been reversed to match %s direction' % direction)

    return df



def all_check(raw_df, direction=None, display=False, distance_type=None, lower_cal=lower_cal, upper_cal=upper_cal):
    """
    Perform quality, calibration and direction check when enable
    :param raw_df: pd.DataFrame()
        Input dataframe
    :param direction: 'EW', 'WE', 'NS', 'SN'
        Defined direction of transect.
    :param display: boolean, default False
         Display rows failing quality check
    :param lower_cal: float, default 0.02 m
        Lower snow depth in meter under which measurement could have been a lower calibration point
    :param upper_cal: float default 1.18 m
        Upper snow depth in meter above which measurement could have been an upper calibration point
    :return output_df: pd.DataFrame()
        Output dataframe
    """
    output_df = raw_df
    if direction is not None:
        output_df = direction_check(output_df, direction=direction)
    else:
        pass
    output_df = reorder(output_df, distance_type=distance_type)
    output_df = quality_check(output_df, display=display)
    output_df = calibration_check(output_df, lower_cal=lower_cal, upper_cal=upper_cal)
    return output_df
