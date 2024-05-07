__name__ = 'magnaprobe_toolbox.load'
__all__ = ['output_filename', 'raw_data', 'qc_data']

import datetime as dt
import logging
import os
import pandas as pd
#import pint
#import pint_pandas as pint
import pyproj

from magnaprobe_toolbox.analysis.distance import compute as compute_distance
from magnaprobe_toolbox.io import output_filename
logger = logging.getLogger(__name__)

# Define
# ureg = pint.UnitRegistry(auto_reduce_dimensions=True)
# pint.set_application_registry(ureg)
#ureg = pint.get_application_registry()


def read_raw(fp, header_row=1):
    """
    Read raw data file from filepath fp.
    :param fp: string
        Filepath to the magnaprobe raw data file.
    :param header_row: int
    :return:
    """
    logger = logging.getLogger(__name__)
    if not fp.endswith('.dat'):
        logger.warning('This may not be an original raw datafile, extension is not ".dat"')
    else:
        pass

    df = pd.read_csv(fp, header=header_row)
    return df


def format_col_headers(input_df):
    """
    Format column headers. We prioritise for sanitation:
    - Lower case with capitalizing first letter of each word
    - Singular form. E.g. Second rather than Seconds
    - Remove "_" spacing. E.g. FixQuality rather than Fix_quality
    - Short abreviation. E.g. Nmbr_satellites: NSatellite

    :param df: pd.DataFrame()
        Input Dataframe containing the raw data, imported with read_row.
    :return:
        Dataframe with formatted and sanitized column headers
    """
    df = input_df.copy()

    # set all columns headers in lower case
    df.columns = [c[0].upper()+c[1:].lower() for c in df.columns]


    # sanitize headers
    header_rename_dict = {'Hdop': 'HDOP', 'Battvolts': 'BattVolts', 'Depthvolts': 'DepthVolts',
                          'Fix_quality': 'FixQuality', 'Nmbr_satellites': 'NSatellite',
                          'Dayofmonth': 'Day', 'Hourofday': 'Hour', 'Minutes': 'Minute',
                          'Seconds': 'Second', 'Microseconds':'Microsecond'}
    header_rename_dict = {k: header_rename_dict[k] for k in header_rename_dict if k in df.columns}
    df.rename(columns=header_rename_dict, inplace=True)

    return df


def remove_junk_row(df):
    """
    Remove junk rows, line 3 and 4 in the original magnaprobe raw datafile

    :param df: pd.DataFrame()
        Dataframe containing the raw data.
    :return: pd.DataFrame()
        Dataframe stripped from the 2 first junk rows.
    """

    df = df.drop(df.index[:2])
    return df


def check_datetime(df, date=dt.date(2020,1,1)):
    """
    Format timestamp to ISO. If timestamps are invalid, artificial timestamps are generated starting at midnight local
    time, with 1-minute increments.

    :param df: pd.DataFrame()
        Dataframe containing the raw data.
    :param date: dt.datetime()
        Date at which the dataset was acquired
    :return:
        Dataframe with formatted timestamp.
    """
    logger = logging.getLogger(__name__)

    try:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='ISO8601')
    except ValueError:
        # Generate artificial timestamp starting at midnight local time
        date = pd.to_datetime(date)
        df['Timestamp'] = pd.date_range(date, periods=len(df), freq="min")
        logger.info(str("Time data is not ISO8601 format. Artificial timestamp is generated starting at midnight local"
                        "time of %s" % date.strftime("%Y%m%d")))
    else:
        pass
    return df


def compute_coordinate(df):
    """
    Compute latitude and longitude coordinate in degree from the integer (_a) and minute (_b) fields. Unnecessary fields
    are then dropped (lat/long -_a, lat/long -_b,'lat/long -ddddd') or renamed (Altitudeb > Altitude)

    :param df: pd.DataFrame()
        Dataframe containing the raw data.
    :return: pd.DataFrame()
        Dataframe containing the latitude, respectively longitude, in the Latitude, respectively Longitude columns,
        given in Decimal Degree.
    """
    df['Latitude'] = df['Latitude_a'].astype('float') + df['Latitude_b'].astype('float')/60
    df['Longitude'] = df['Longitude_a'].astype('float') + df['Longitude_b'].astype('float')/60

    # drop unnecessary coordiante axis
    df.drop(['Latitude_a', 'Latitude_b', 'Longitude_a', 'Longitude_b', 'Latitudeddddd', 'Longitudeddddd'], axis=1, inplace=True)
    df.rename({'Altitudeb': 'Altitude'}, axis=1, inplace=True)
    # # set units
    # df['Latitude'] = df['Latitude'].astype('pint[degree]')
    # df['Longitude'] = df['Longitude'].astype('pint[degree]')
    return df


def convert_wgs_to_utm(df, local_EPSG=4326):
    """
    Convert latitude/longitude degree coordinates from WSG84 into local projection (AK EPSG 3338; Web Mercator
    Projection 3857; WGS84: 4326).
    By default, altitude Z=0.

    :param df: pd.DataFrame()
        Dataframe containing the raw data.
    :param local_EPSG: int, default 4326
        EPSG Geodetic parameter dataset to convert to. Default is 4326 for WGS84
    :return: pd.DataFrame()
        Dataframe augmented with the columns X, Y and Z containing the latitude/longitude coordinate projected locally
        in m.
    """
    if isinstance(local_EPSG, float):
        if local_EPSG == int(local_EPSG):
            local_EPSG = int(local_EPSG)
        else:
            logger.warning('EPSG number is not an integer. Using default 4326 instead')
    else:
        pass
    # TODO: include elevation Z
    # Create X, Y geospatial in ESPG 3338 reference for Alaska
    xform = pyproj.Transformer.from_crs(4326, local_EPSG)
    df['X'], df['Y'] = xform.transform(df['Latitude'], df['Longitude'])
    df['Z'] = [0.]*len(df)

    # # set units
    # df['X'] = df['X'].astype('pint[m]')
    # df['Y'] = df['Y'].astype('pint[m]')
    # df['Z'] = df['Z'].astype('pint[m]')
    return df



def convert_snowdepth_to_m(df):
    """
    Convert snow depth from cm to m. Remove unnecessary `Snowcm` column field.

    :param df: pd.DataFrame()
        Dataframe containing the raw data.
    :return: pd.DataFrame()
        Dataframe containing the snow depth given only in meter (m)
    """
    df['SnowDepth'] = df['Depthcm'].astype(float) / 100.0
    df.drop(columns=['Depthcm'], inplace=True)
    # # set units
    # df['SnowDepth'] = df['SnowDepth'].astype('pint[m]')
    return df


def raw_data(fp, local_EPSG=4326):
    """
    Load and clean raw data from filepath by computing local UTM coordinate, removing unnecessary headers and sanitizing
    the other headers

    :param fp: string
        Input data filepath
    :param local_EPSG: int, default 4326
        EPSG Geodetic parameter dataset to convert to. Default is 4326 for WGS84
    :return df: pd.DataFrame()
        Dataframe containing the clean raw data
    """
    df = read_raw(fp)
    df = format_col_headers(df)
    df = remove_junk_row(df)
    df = check_datetime(df)
    df = convert_snowdepth_to_m(df)
    df = compute_coordinate(df)
    df = convert_wgs_to_utm(df, local_EPSG)
    df = compute_distance(df)
    return df.reset_index(drop=True)


def qc_data(fp):
    """
    Read quality-checked data file containing magnaprobe data
    :param fp: string
        Filepath to the magnaprobe quality-checked data file.
    :return df: pd.DataFrame()
        DataFrame containing quality-checked data
    """
    if not fp.endswith('.csv'):
        logger.warning('This may not be a QC-ed datafile, extension does not match.')
    else:
        pass

    df = pd.read_csv(fp, header=[0])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df
