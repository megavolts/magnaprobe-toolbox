import pandas as pd
import numpy as np
from scipy.ndimage import convolve
import logging

logger = logging.getLogger(__name__)

def binomial_1Dkernel(n_order):
    """
    1D kernel for binomial filter of order n, with the sum of the coefficient equal to 1.

    :param n_order: int
        Order of the binomial filter
    :return: 1darray
        A 1d array of dimension n_order+1 containing the binomial filter coefficient.
    """
    bi = np.asarray([1])
    for _ in range(n_order):
        bi = np.convolve([0.5, 0.5], bi)
    return bi / bi.sum()


def apply(df, n_order, mode='reflect', fillna='interp'):
    """
    Smooth snow depth along the transect using a n-order binomial filter
    :param df: 2darray
        The input array.
    :param n_order: int
        Order of the binomial filter
    :param mode: {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}, optional
        The mode parameter determines how the input array is extended beyond its boundaries. Default is ‘reflect’. Behavior for each valid value is as follows:
        ‘reflect’ (d c b a | a b c d | d c b a)
            The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.
        ‘constant’ (k k k k | a b c d | k k k k)
            The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.
        ‘nearest’ (a a a a | a b c d | d d d d)
            The input is extended by replicating the last pixel.
        ‘mirror’ (d c b | a b c d | c b a)
            The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.
        ‘wrap’ (a b c d | a b c d | a b c d)
            The input is extended by wrapping around to the opposite edge.
    :param merge: boolean, optional
        If merge is true, the smooth data are merge with the original data
    :return: pd.DataFrame()
        A DataFrame containing an additional column named "n_order" in which is stored the order of the filter applied
        If merge is true, the dataframe contains the smooth and original data
        If merge is false, the dataframe contains only the smooth data
    """
    output_df = df.copy()

    if fillna == 'interp':
        df['SnowDepth'] = df['SnowDepth'].interpolate()
        logger.warning('Interpolating over NaN snow depth value')
    else:
        pass
    output_df['SnowDepth'] = convolve(df['SnowDepth'], binomial_1Dkernel(n_order), mode=mode)
    output_df.loc[:, 'n_order'] = n_order
    return output_df
