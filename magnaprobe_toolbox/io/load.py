__name__ = 'magnaprobe_toolbox.load'
__all__ = ['output_filename', 'raw_data', 'qc_data']

import datetime as dt
import logging
import os
import pandas as pd
#import pint
#import pint_pandas as pint
import pyproj

from magnaprobe_toolbox.io.tools import compute_distance
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


def format_col_headers(df):
    """
    Format all columns header in lower case, but the first letter.
    :param fp: pd.DataFrame()
        Dataframe containing the raw data, imported with read_row.
    :return:
        Dataframe with formatted column headers
    """
    # set all columns headers in lower case
    df.columns = [c[0].upper()+c[1:].lower() for c in df.columns]
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
    time, with 1-second increments.

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
        df['Timestamp'] = pd.date_range(date, periods=len(df), freq="s")
        logger.info(str("Time data is not ISO8601 format. Artificial timestamp is generated starting at midnight local"
                        "time of %s" % date.strftime("%Y%m%d")))
    else:
        pass
    return df


def compute_coordinate(df):
    """
    Compute latitude and longitude coordinate in degree from the integer (_a) and minute (_b) fields.

    :param df: pd.DataFrame()
        Dataframe containing the raw data.
    :return: pd.DataFrame()
        Dataframe containing the latitude, respectively longitude, in the Latitude, respectively Longitude columns,
        given in Decimal Degree.
    """
    df['Latitude'] = df['Latitude_a'].astype('float') + df['Latitude_b'].astype('float')/60
    df['Longitude'] = df['Longitude_a'].astype('float') + df['Longitude_b'].astype('float')/60

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
    :param local_EPSG: string, default '4326'
        EPSG Geodetic parameter dataset to convert to. Default is 4326 for WGS84
    :return: pd.DataFrame()
        Dataframe augmented with the columns X, Y and Z containing the latitude/longitude coordinate projected locally
        in m.
    """
    if isinstance(local_EPSG, (int, float)):
        local_EPSG=str(int(local_EPSG))
    else:
        pass
    # TODO: include elevation Z
    # Create X, Y geospatial in ESPG 3338 reference for Alaska
    xform = pyproj.Transformer.from_crs('4326', local_EPSG)
    df['X'], df['Y'] = xform.transform(df['Latitude'], df['Longitude'])
    df['Z'] = [0.]*len(df)

    # # set units
    # df['X'] = df['X'].astype('pint[m]')
    # df['Y'] = df['Y'].astype('pint[m]')
    # df['Z'] = df['Z'].astype('pint[m]')
    return df



def convert_snowdepth_to_m(df):
    """
    Convert snow depth from cm to m.

    :param df: pd.DataFrame()
        Dataframe containing the raw data.
    :return: pd.DataFrame()
        Dataframe containing the snow depth given in meter (m).
    """
    df['SnowDepth'] = df['Depthcm'].astype(float) / 100.0
    df['SnowTension'] = df['Depthvolts'].astype(float)

    # # set units
    # df['SnowDepth'] = df['SnowDepth'].astype('pint[m]')
    # df['SnowTension'] = df['SnowTension'].astype('pint[V]')
    return df


def raw_data(fp, local_EPSG=4326):
    """
    Load and clean raw data from filepath, and compute local UTM coordinate

    :param fp: string
        Data filepath
    :param local_EPSG: string, default '4326'
        EPSG Geodetic parameter dataset to convert to. Default is 4326 for WGS84

    :return: pd.DataFrame()

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
    Read qc data file from filepath raw_fp.
    :param df: string
        Filepath to the magnaprobe raw data file.
    :param header_row: int
    :return:
    """
    if not fp.endswith('.csv'):
        logger.warning('This may not be a QC-ed datafile, extension does not match.')
    else:
        pass

    df = pd.read_csv(fp, header=[0])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df
