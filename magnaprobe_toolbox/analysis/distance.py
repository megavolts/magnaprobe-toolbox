__all__ = ['compute']
__author__ = 'Marc Oggier'
import logging
import pandas as pd
logger = logging.getLogger(__name__)



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


def compute(input_df, origin_pt=None):
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
    output_df = input_df.copy()

    if origin_pt is None:
        x0 = input_df.iloc[0]['X']
        y0 = input_df.iloc[0]['Y']
        z0 = input_df.iloc[0]['Z']
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

    # TODO: if previous point is missing, compute with the last known coordiante, aka filter nas row out
    # USE 2040417-ICE
    output_df['TrackDist'] = euclidian_distance(output_df['X'], output_df['X'].shift(), output_df['Y'], output_df['Y'].shift())
    output_df.iloc[0, output_df.columns.get_loc('TrackDist')] = 0
    # Compute cumulative sum of distance between two consecutive points
    output_df['TrackDistCum'] = output_df['TrackDist'].cumsum()
    output_df.iloc[0, output_df.columns.get_loc('TrackDistCum')] = 0
    # Compute distance between the origin point and the current point
    output_df['DistOrigin'] = euclidian_distance(output_df['X'], x0, output_df['Y'], y0)

    return output_df