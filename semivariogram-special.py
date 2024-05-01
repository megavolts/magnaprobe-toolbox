from cmcrameri import cm
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os
import scipy.stats as stats

# Magnaprobe calibration value
lower_cal = 0.02  # m
upper_cal = 1.18  # m

plot_origin_flag = False

raw_file = '/home/megavolts/git/magnaprobe-toolbox/data/SemiVariogramData_ForMarc.xlsx'

# Generate output file
header_row = 0
raw_df = pd.read_excel(raw_file, header=header_row)
raw_bkp = pd.read_excel(raw_file, header=header_row)
# Inspired by https://www.wavemetrics.net/doc/igorman/III-07%20Analysis.pdf
def wavestats(Y):
    hs_stats = {}
    hs_stats['npts'] = len(Y)
    hs_stats['sum'] = np.sum(Y)
    hs_stats['avg'] = stats.tmean(Y)
    hs_stats['med'] = np.median(Y)
    hs_stats['max'] = stats.tmax(Y)
    hs_stats['maxloc'] = np.where(Y == hs_stats['max'])[0]
    hs_stats['min'] = stats.tmin(Y)
    hs_stats['minloc'] = np.where(Y == hs_stats['min'])[0]
    hs_stats['sdev'] = stats.tstd(Y)  # Standard deviation
    hs_stats['adev'] = np.sum(np.abs((Y - np.mean(Y))))/len(Y)
    hs_stats['sem'] = stats.tsem(Y)  # Standard error of mean
    hs_stats['rms'] = np.sqrt(np.sum(Y**2)/len(Y))  # Root mean square
    hs_stats['skew'] = stats.skew(Y, bias=False)
    hs_stats['kurt'] = stats.kurtosis(Y, bias=False)
    return hs_stats

raw_df['SnowDepth'] = raw_df['AA']

# Compute wave statistic for snow depth
hs_stat = wavestats(raw_df['SnowDepth'])

# Compute snow depth histogram
hs_max = 1.2
hs_min = 0
d_hs = 0.1
n_bins = int(np.round((hs_max - hs_min) / d_hs + 1))
counts, bins = np.histogram(raw_df['SnowDepth'], np.arange(hs_min, hs_max+d_hs/2, d_hs), density=True)
hs_bin_mid = bins[:-1] + np.diff(bins)/2

# For log normal to snow depth distribution
pdf_p = stats.lognorm.fit(raw_df['SnowDepth'])
hs_x = np.arange(hs_min, hs_max+0.001, 0.001)
hs_fitted_ln = stats.lognorm(*pdf_p)

#Wavestats $A_wave
hs_stat = wavestats(raw_df['SnowDepth'])
#mean_depth = num2str(V_avg)
mean_depth = hs_stat['avg']
#depth_dev = num2str(V_sdev)
depth_dev = hs_stat['sdev']
#npoints = num2str(V_npnts)
npoints = hs_stat['npts']


# Binomial filter
def binomial_1Dkernel(n_order):
    """
    1D kernel for binomial filter of order n, with the sum of the coefficient equal to 1.

    :param n_order: int
        Order of the filter
    :return: 1darray
        A 1d array of dimension n_order+1 containing the binomial filter coefficient.
    """
    bi = np.asarray([1])
    for _ in range(n_order):
        bi = np.convolve([0.5, 0.5], bi)
    return bi / bi.sum()


from scipy.ndimage import convolve
# apply Gaussian filterin
plt.figure()
plt.plot(raw_df['SnowDepth'], color='b',  label='AA', alpha=0.6)
plt.plot(raw_bkp['AA'], color='b', linestyle=':', alpha=0.6, label='AA')
plt.plot(convolve(raw_df['SnowDepth'], binomial_1Dkernel(4), mode='reflect'), color='orange', label='order =1')
plt.plot(raw_bkp['BB'], color='orange', linestyle=':', alpha=0.6, label='BB')
plt.plot(convolve(raw_df['SnowDepth'], binomial_1Dkernel(19), mode='reflect'), color='g', label='$\sigma$=2')
plt.plot(raw_bkp['CC'], color='g', linestyle=':', alpha=0.6, label='CC')
plt.plot(convolve(raw_df['SnowDepth'], binomial_1Dkernel(49), mode='reflect'), color='r', label='$\sigma$=3')
plt.plot(raw_bkp['DD'], color='r', linestyle=':', alpha=0.6, label='DD')
plt.legend()
plt.xlim([0, 500])
plt.savefig('/home/megavolts/Desktop/AA_Gaussian_MO.pdf')
plt.show()

# Range of wave to consider
v_npnts = raw_df['SnowDepth'].__len__()
# duplicate/O/R=(xcsr(A),xcsr(B))$A_wave B_wave, C_wave, D_wave, semivar
A_wave = raw_df['SnowDepth']
B_wave = raw_df['SnowDepth']
C_wave = raw_df['SnowDepth']
D_wave = raw_df['SnowDepth']
semivar = []


def semivariogram(x):



# 	wavestats/Q B_wave
B_stat = wavestats(B_wave)
orig_length = v_npnts
N = orig_length
half = v_npnts / 2


lag = 0
while(lag <= (N / 2 + 2)):
    print(lag)
    lag += 1
    endpoint = orig_length - lag
    #C_wave = B_wave[p + lag]
    C_wave = A_wave.iloc[lag:].reset_index(drop=True)
    # DeletePoints endpoint, 1, D_wave
#    D_wave = D_wave[-1][:-1]
    # DeletePoints endpoint, 1, C_wave
#    C_wave = C_wave[-1][:-1]
    # D_wave = ((B_wave[p] - C_wave[p]) ^ 2) / (2 * endpoint)
    B_wave = A_wave[:-lag].reset_index(drop=True)
    D_wave = (B_wave - C_wave)**2 / (2*endpoint)
    # wavestats/Q D_wave
    D_stat = wavestats(D_wave)
    # gamma = v_avg * v_npnts
    gamma = D_stat['avg'] * D_stat['npts']
    # semivar[lag] = gamma
    semivar.append([lag, gamma])

semivar=np.array(semivar)

plt.figure()
plt.plot(semivar.transpose()[0],semivar.transpose()[1])
plt.xlim([0, 250])
plt.ylim([0, 50])
plt.show()
# y, x = np.mgrid[0:500:5, 0:500:5]
# coords = np.c_[y.ravel(), x.ravel()]
# values = z[y.ravel(), x.ravel()]

# Calculate Variogram
plt.figure()
lag = 0
while(lag <=101):
    lag = lag+10
    V = skg.Variogram(coords, values, n_lags = lag, bin_func = 'even', maxlag = 200)
    vals = np.array(V.get_empirical())
    plt.scatter(vals[0], vals[1])
plt.show()


# Plot basic figures:
site = output_fn.split('_')[1]
date = output_fn.split('.')[0][-8:]
if '_longline' in output_fn:
    name = 'Long transect'
elif '_line' in output_fn:
    name = '200 m transect'
elif 'library' in output_fn:
    name = 'Library site'

# Data status plot
# Move origin points to x0, y0
if plot_origin_flag:
    raw_df['x0'] = raw_df['X'] - raw_df.iloc[0]['X']
    raw_df['y0'] = raw_df['Y'] - raw_df.iloc[0]['Y']
else:
    raw_df['x0'] = raw_df['X']
    raw_df['y0'] = raw_df['Y']
snow_depth_scale = raw_df['SnowDepth'].max()-raw_df['SnowDepth'].min()
z0 = cm.davos(raw_df['SnowDepth']/snow_depth_scale)
plt.style.use('ggplot')

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
