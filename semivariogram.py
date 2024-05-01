from cmcrameri import cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.ndimage import convolve

logger = logging.getLogger(__name__)

plot_origin_flag = False
data_fp = '/mnt/data/UAF-data/working_a/SALVO/20240416-ARM/magnaprobe/salvo_arm_longline_magnaprobe-geo1_20240416.a3.csv'


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


def stat_basics(y_data):
    """
    Compute basic statistic (sum, average, median, maximum, minimum, minimum location, maximum location
    standard deviation, root mean squared, skewness, kurtosis)
    :param Y: array, pd.Series(), pd.DataFrame()
    :return: dict
        Dictionnary containing basic statistic for the array
    """
    hs_stats = {}
    hs_stats['npts'] = len(y_data)
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


data_df = import_qc_data(data_fp)

def semivariogram(input_df):
    """
    Compute the semivariogram of the input dataframe
    :param input_df: pd.DataFrame()
        The input dataframe, in which the index is the geolocation of the metric to analyse
    :return semivr_df: pd.DataFrame()
        A dataframe containing the semivariogram data, including a column in which is store the order of filtering
    """
# Range of wave to consider
# duplicate/O/R=(xcsr(A),xcsr(B))$A_wave B_wave, C_wave, D_wave, semivar
# B_wave = raw_df.iloc[0:v_npnts][['TrackDistCum', 'SnowDepth']]
# C_wave = raw_df.iloc[0:v_npnts][['TrackDistCum', 'SnowDepth']]
# D_wave = raw_df.iloc[0:v_npnts][['TrackDistCum', 'SnowDepth']]
# semivar = []
# # 	wavestats/Q B_wave
# B_stat = wavestats(B_wave[-1])
# orig_length = v_npnts
# N = orig_length
# half = v_npnts / 2
    semivar = np.array([])
    lag = 0
    while(lag <= (len(input_df) / 2 + 2)):
        lag += 1
        # endpoint = len(data_df['SnowDepth']) - lag
        #C_wave = B_wave[p + lag]
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


def binomial_filter(df, n_order, mode='reflect'):
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
    output_df['SnowDepth'] = convolve(df['SnowDepth'], binomial_1Dkernel(n_order), mode=mode)
    output_df.loc[:, 'n_order'] = n_order
    return output_df


def compute_semivariogram(input_df, n_order_l, mode='reflect'):
    """
   The semivariogram of the input is computed after applying a binomial filter of order n.

    :param input_df: pd.DataFrame()
        The input DataFrame, in which the index is the geolocation of the metric to analyse
    :param n_order_l: int, list of int
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
    if isinstance(n_order_l, (int, float)):
        n_order_l = np.array([n_order_l])
    elif isinstance(n_order_l, list):
        n_order_l = np.array([n_order_l])
    else:
        pass
    if 0 not in n_order_l:
        n_order_l = np.append([0], n_order_l)

    smooth_df = pd.DataFrame()
    semivr_df = pd.DataFrame()
    for n_order in n_order_l:
        _temp_df = input_df.copy()
        if n_order == 0:
            _temp_df = input_df
            _temp_df.loc[:, 'n_order'] = [0] * len(_temp_df)
        else:
            _temp_df = binomial_filter(input_df, n_order=n_order, mode='reflect')
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
    return smooth_df, semivr_df

input_df = data_df[['TrackDistCum', 'SnowDepth']].set_index('TrackDistCum')
smooth_data, hs_vs = compute_semivariogram(input_df, [5, 20, 50])



# Figure
nrows, ncols = 2, 1
w_fig, h_fig = 8, 11
fig = plt.figure(figsize=[w_fig, h_fig])
gs1 = gridspec.GridSpec(nrows, ncols, height_ratios=[1, 1], width_ratios=[1])
ax = [[fig.add_subplot(gs1[0, 0])], [fig.add_subplot(gs1[1, 0])]]
ax = np.array(ax)

for n_order in [0, 5, 20, 50]:
    st_plot_df = smooth_data.loc[smooth_data.n_order == n_order]
    hs_plot_df = hs_vs.loc[hs_vs.n_order == n_order]
    ax[0, 0].plot(st_plot_df.index, st_plot_df['SnowDepth'])

    ax[1, 0].plot(hs_plot_df.index, hs_plot_df['Semivariogram'])

plt.show()

# Define figure subplots
nrows, ncols = 4, 1
w_fig, h_fig = 8, 11
fig = plt.figure(figsize=[w_fig, h_fig])
gs1 = gridspec.GridSpec(nrows, ncols, height_ratios=[1, 1, 1, 1], width_ratios=[1])
ax = [[fig.add_subplot(gs1[0, 0])], [fig.add_subplot(gs1[1, 0])], [fig.add_subplot(gs1[2, 0])], [fig.add_subplot(gs1[3, 0])]]
ax = np.array(ax)
ax[0, 0].plot(raw_df['TrackDistCum'], raw_df['SnowDepth'], color='steelblue')
ax[0, 0].fill_between(raw_df['TrackDistCum'], raw_df['SnowDepth'], [0]*len(raw_df), color='lightsteelblue')

ax[0, 0].set_xlabel('Distance along the transect (m)')
ax[0, 0].set_ylabel('Snow Depth (m)')
ax[0, 0].set_xlim([0, raw_df['TrackDistCum'].max()])
ax[0, 0].set_ylim([0, 1.2])

ax[1, 0].scatter(raw_df['x0'], raw_df['y0'], c=z0)
if plot_origin_flag:
    ax[1, 0].set_xlabel('X coordinate (m)')
    ax[1, 0].set_ylabel('Y coordinate (m)')
else:
    ax[1, 0].set_xlabel('UTM X coordinate (m)')
    ax[1, 0].set_ylabel('UTM Y coordinate (m)')

# Snow depth probability function:
ax[2, 0].bar(hs_bin_mid, counts, 0.1, edgecolor='steelblue', color='lightsteelblue')
ax[2, 0].plot(hs_x , hs_fitted_ln.pdf(hs_x), 'k')
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
