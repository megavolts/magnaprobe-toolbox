import numpy as np
import pandas as pd
from magnaprobe_toolbox.analysis import binomial_filter
import magnaprobe_toolbox as mt
__all__ = ['semivariogram']

def semivariogram_ms(input_df):
    """
    Compute the semivariogram of the input dataframe
    :param input_df: pd.DataFrame()
        The input dataframe, in which the index is the geolocation of the metric to analyse
    :return semivr_df: pd.DataFrame()
        A dataframe containing the semivariogram data, including a column in which is store the order of filtering
    """
    # Range of wave to consider
    A_wave=input_df['SnowDepth'].to_numpy()
    # duplicate/O/R=(xcsr(A),xcsr(B))$A_wave B_wave, C_wave, D_wave, semivar
    B_wave = A_wave.copy()
    C_wave = A_wave.copy()
    D_wave = A_wave.copy()
    semivar = A_wave.copy()
    # semivar=NaN
    semivar = [np.nan]*len(A_wave)
    # wavestats/Q B_wave
    B_stat = mt.analysis.statistic.basic(B_wave)
    # orig_lenght=v_npts
    orig_length = B_stat['N']
    # N=orig_length
    N = orig_length
    # half=v_npts/2
    half = B_stat['N']/2
    lag = 0
    while(lag <= (N / 2 + 2)):
        # lag=lag+1
        lag += 1
        # endpoint =orig_length-lag
        endpoint = orig_length - lag
        # C_wave = B_wave[p + lag]
        # lag-shifted snow depth array[lag:]
        C_wave[0:endpoint] = A_wave[lag:]
        # DeletePoints endpoint, 1, D_wave
        D_wave = D_wave[:-1]
        # DeletePoints endpoint, 1, C_wave
        C_wave = C_wave[:-1]
        # D_wave = ((B_wave[p] - C_wave[p]) ^ 2) / (2 * endpoint)
        # need to shortent B_wave for python
        B_wave = B_wave[:-1]
        D_wave = ((B_wave - C_wave)**2)/(2*endpoint)
        # wavestats/Q D_wave
        D_stat = mt.analysis.statistic.basic(D_wave)
        # gamma = v_avg * v_npnts
        gamma = D_stat['avg'] / D_stat['N']
        # semivar[lag] = gamma
        semivar[lag] = gamma
    semivar[0] = 0
    semivar = np.array(semivar)
    semivar = semivar.transpose()
    output_df = pd.DataFrame(semivar, columns=['Semivariogram'], index=input_df.index)
    output_df.index.name = 'Dist'
    return output_df


def semivariogram(input_df):
    """
    Compute the semivariogram of the input dataframe
    :param input_df: pd.DataFrame()
        The input dataframe, in which the index is the geolocation of the metric to analyse
    :return semivr_df: pd.DataFrame()
        A dataframe containing the semivariogram data, including a column in which is store the order of filtering
    """
    # Range of wave to consider
    ## A_wave=input_df
    ## A_wave = raw_df[['TrackDistCum', 'SnowDepth']].set_index('TrackDistCum', drop=True)
    # duplicate/O/R=(xcsr(A),xcsr(B))$A_wave B_wave, C_wave, D_wave, semivar
    ## B_wave = A_wave
    ## C_wave = A_wave
    ## D_wave = A_wave
    ## semivar = A_wave
    # semivar=NaN
    ## semivar['SnowDepth'] = np.nan
    # wavestats/Q B_wave
    # B_stat = wavestat(B_wave)
    # orig_length = len(A_wave)
    # N = orig_length
    # half = len(A_wave) / 2
    semivar = np.array([])
    lag = 0
    while(lag <= (len(input_df) / 2 + 2)):
        # lag=lag+1
        lag += 1
        # endpoint =orig_length-lag
        ## endpoint = orig_length - lag
        # C_wave = B_wave[p + lag]
        # lag-shifted snow depth array[lag:]
        lagged_df = input_df.copy().iloc[lag:].reset_index(drop=True)
        # C_wave = A_wave.iloc[lag:].reset_index(drop=True)
        # DeletePoints endpoint, 1, D_wave
    #    D_wave = D_wave[-1][:-1]
        # DeletePoints endpoint, 1, C_wave
    #    C_wave = C_wave[-1][:-1]
        # D_wave = ((B_wave[p] - C_wave[p]) ^ 2) / (2 * endpoint)
        # shorten snow depth array [:-lag]
        short_df = input_df.copy().iloc[:-lag].reset_index(drop=True)
    #    B_wave = A_wave[:-lag].reset_index(drop=True)
        diff_df = (short_df - lagged_df)**2 / (2*len(short_df))
        # wavestats/Q D_wave
#        lag_stat = stat_basics(hs_dif)
        # gamma = v_avg * v_npnts
        gamma = diff_df.mean().values * len(diff_df)
        # semivar[lag] = gamma
        lag_dist = input_df.index[lag]
        if len(semivar) == 0:
            semivar = np.array([[lag_dist, gamma[0]]])
        else:
            semivar = np.vstack([semivar, [lag_dist, gamma[0]]])
    semivar = semivar.transpose()
    output_df = pd.DataFrame(semivar[1], columns=['Semivariogram'], index=semivar[0])
    output_df.index.name = 'Dist'
    return output_df



def compute(input_df, n_orders, mode='reflect'):
    """
   The semivariogram of the input is computed after applying a binomial filter of order n.

    :param input_df: pd.DataFrame()
        The input DataFrame, in which the index is the geolocation of the metric to analyse
    :param n_orders: int, list of int
        A list of number corresponding to the orders of the binomial filter to apply.
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
    :return smooth_df: pd.DataFrame()
        A dataframe containing the smoothen data, including a column in which is store the order of filtering
    :return semivr_df: pd.DataFrame()
        A dataframe containing the semivariogram data, including a column in which is store the order of filtering
    """
    if isinstance(n_orders, (int, float)):
        n_orders = np.array([n_orders])
    elif isinstance(n_orders, list):
        n_orders = np.array(n_orders)
    else:
        pass
    if 0 not in n_orders:
        n_orders = np.append([0], n_orders)

    if len(input_df.shape) < 2:
        input_df = pd.DataFrame(input_df)

    smooth_df = pd.DataFrame()
    semivr_df = pd.DataFrame()
    for n_order in n_orders:
        _temp_df = input_df.copy()
        if n_order == 0:
            _temp_df['n_order'] = [0] * len(_temp_df)
        else:
            _temp_df = binomial_filter.apply(input_df, n_order=n_order, mode='reflect')
        if smooth_df.empty:
            smooth_df = _temp_df
        else:
            smooth_df = pd.concat([smooth_df, _temp_df], join='outer')

        _temp_df = semivariogram(_temp_df.drop(columns=['n_order']))
        _temp_df.loc[:, 'n_order'] = n_order
        if semivr_df.empty:
            semivr_df = _temp_df
        else:
            semivr_df = pd.concat([semivr_df, _temp_df], join='outer')
    return smooth_df.sort_index(), semivr_df.sort_index()
