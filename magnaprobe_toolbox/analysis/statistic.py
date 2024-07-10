import pandas as pd
import numpy as np
import scipy.stats as stats
import logging

logger = logging.getLogger(__name__)

def basic(hs):
    """
    Compute basic statistic (sum, average, median, maximum, minimum, minimum location, maximum location
    standard deviation, root mean squared, skewness, kurtosis)
    :param hs: list, ndarray
        The input array containing snow depth
    :return: dict
        Dictionary containing basic statistic for the array
    """
    if isinstance(hs, list):
        hs = np.array(hs)

    # Remove negative value:
    hs = hs[hs >= 0]

    # remove nan value
    hs_stats = {}
    hs_stats['N'] = len(hs)

    hs = hs[~np.isnan(hs)]
    hs_stats['Nnan'] = hs_stats['N'] - len(hs)
    hs_stats['sum'] = np.sum(hs)
    hs_stats['avg'] = np.mean(hs)
    hs_stats['mu'] = np.mean(hs)
    hs_stats['mean'] = np.mean(hs)
    hs_stats['med'] = np.median(hs)
    hs_stats['max'] = np.max(hs)
#    hs_stats['maxloc'] = np.where(hs == hs_stats['max'])[0]
    hs_stats['min'] = np.min(hs)
#    hs_stats['minloc'] = np.where(hs == hs_stats['min'])[0]
    hs_stats['sdev'] = stats.tstd(hs)  # Standard deviation
    hs_stats['std'] = hs_stats['sdev']  # Standard deviation
    hs_stats['sigma'] = hs_stats['sdev']  # Standard deviation
    hs_stats['adev'] = np.sum(np.abs((hs - np.mean(hs))))/len(hs)
    hs_stats['sem'] = stats.tsem(hs)  # Standard error of mean
    hs_stats['rms'] = np.sqrt(np.sum(hs**2)/len(hs))  # Root mean squared
    hs_stats['skew'] = stats.skew(hs, bias=False)
    hs_stats['kurt'] = stats.kurtosis(hs, bias=False)

    return hs_stats

def histogram(input_df, bins=None):
    """
    Compute histogram of input. If bins are not defined,
    :param input_df: pd.DataFrame()
        The input DataFrame, in which the index is the geolocation of the metric to analyse
    :param bins: int or sequence of scalars or str, optional
        If bins is an int, it defines the number of equal-width bins in the given range (10, by default). If bins is a
        sequence, it defines a monotonically increasing array of bin edges, including the rightmost edge, allowing for
        non-uniform bin widths.
        If bins is a string, it defines the method used to calculate the optimal bin width, as defined by
        histogram_bin_edges.
    :return output_df: pd.DataFrame()
        Dictionnary containing histogram data bin_mid, count
    """
    if bins is None:
        counts, bins = np.histogram(input_df, density=True)
    else:
        counts, bins = np.histogram(input_df, bins, density=True)
    hy_bin_mid = bins[:-1] + np.diff(bins) / 2
    output_df = pd.DataFrame(data=counts, index=hy_bin_mid, columns=['Count'])
    output_df.index.name = 'bin_mid'
    return output_df


def pdf(input_df):
    """
    Compute probability density function for input_df

    :param input_df: pd.DataFrame()
        The input DataFrame, in which the index is the geolocation of the metric to analyse
    :return pdf_f: function object
        Fitted lognorm probability density function for the input. Call with y=pdf_f.pdf(x)
     lognorm probability density function for the input
    """
    pdf_p = stats.lognorm.fit(input_df)
    pdf_f = stats.lognorm(*pdf_p)
    return pdf_f
