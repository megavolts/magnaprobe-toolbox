import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from matplotlib import gridspec as gridspec

from magnaprobe_toolbox import analysis
from magnaprobe_toolbox.io import upper_cal, lower_cal

def set_origin(plot_df, plot_from_origin=False):
    if plot_from_origin:
        plot_df["x0"] = plot_df["X"] - plot_df.iloc[0]["X"]
        plot_df["y0"] = plot_df["Y"] - plot_df.iloc[0]["Y"]
    else:
        plot_df["x0"] = plot_df["X"]
        plot_df["y0"] = plot_df["Y"]
    return plot_df

def stat_annotation(hs_stats, stat_l=['N', 'min', 'max', 'mean', 'std']):
    """
    :param hs_stats: pd.DataFrame()
        Dataframe containing the statistic values to display
    :param stat_l: list of string
        List containing the statisitc metric to display
    :return:

    """
    statstr = ''
    for stat in stat_l:
        statval = hs_stats[stat]
        if stat in 'mu':
            statstr += '$\mu$ =' + '%.2f\n' %statval
        if stat in 'sigma':
            statstr += '$\sigma$ =' + '%.2f\n' %statval
        if stat in ['N', 'npts']:
            statstr += '$N =$' + '%.0d\n' %statval
        else:
            statstr += stat + ' = ' + '%.2f\n' %statval
    statstr = statstr[:-1]
    statprops = dict(boxstyle='round', facecolor='lightsteelblue', alpha=0.5, edgecolor='steelblue')
    return statstr, statprops


def summary(input_df, fig_fp=None, hist=True, sv_order=[0, 5, 20, 50], library='True'):
    """


    :param input_df:
    :param fig_fp: str
        Filepath to save the figure. If None, figure is not save
    :return fig: matplotlib.pyplot.figure()

    """
    # Define figure subplots
    ncols = 1
    nrows = 2

    hs_df = input_df.loc[input_df['SnowDepth'].notnull(), ['SnowDepth']]

    # Compute snow depth statistic
    hs_stats = analysis.statistic.basic(input_df['SnowDepth'].to_numpy())
    statstr, statbox = stat_annotation(hs_stats)

    if library:
        sv_order = None

    if hist:
        nrows += 1

        # Compute snow depth histogram for
        hs_min = 0
        hs_max = 1.2
        dhs = 0.05
        bins = np.arange(hs_min, hs_max + dhs / 2, dhs)
        hs_hist = analysis.statistic.histogram(hs_df, bins)

        # Compute snow depth pdf
        hs_pdf_fit = analysis.statistic.pdf(hs_df)
        hs_x = np.arange(lower_cal, upper_cal, dhs/10)

        # Plot
        snow_depth_scale = hs_df.max().values - hs_df.min().values
        hs0 = cm.davos(input_df.loc[input_df['SnowDepth'].notnull(), ['SnowDepth']]/snow_depth_scale[0])
        x0 = input_df.loc[input_df['SnowDepth'].notnull(), ['x0']]
        y0 = input_df.loc[input_df['SnowDepth'].notnull(), ['y0']]

    if sv_order is not None:
        nrows += 1
        smooth_data, hs_vs = analysis.semivariogram.compute(input_df['SnowDepth'], [5, 20, 50])



    w_fig, h_fig = 8, 11
    fig = plt.figure(figsize=[w_fig, h_fig])
    gs1 = gridspec.GridSpec(4, ncols, height_ratios=[1]*4, width_ratios=[1])
    ax = [[fig.add_subplot(gs1[0, 0])], [fig.add_subplot(gs1[1, 0])], [fig.add_subplot(gs1[2, 0])], [fig.add_subplot(gs1[3, 0])]]
    ax = np.array(ax)
    if library:
        ax[0, 0].scatter(input_df.index, input_df['SnowDepth'], color='steelblue')

    else:
        ax[0, 0].plot(input_df.index, input_df['SnowDepth'], color='steelblue')
        ax[0, 0].fill_between(input_df.index, input_df['SnowDepth'], [0]*len(input_df), color='lightsteelblue')

    ax[0, 0].set_xlabel('Distance along the transect (m)')
    ax[0, 0].set_ylabel('Snow Depth (m)')
    ax[0, 0].text(0.8, 0.9, statstr, bbox=statbox, transform=ax[0, 0].transAxes,
            fontsize=10, verticalalignment='top')
    ax[0, 0].set_xlim([0, input_df.index.max()])
    ax[0, 0].set_ylim([0, 1.2])


    ax[1, 0].scatter(x0, y0, c=hs0)
    if input_df['x0'][0] == 0 and input_df['y0'][0] == 0:
        ax[1, 0].set_xlabel('X coordinate (m)')
        ax[1, 0].set_ylabel('Y coordinate (m)')
    else:
        ax[1, 0].set_xlabel('UTM X coordinate (m)')
        ax[1, 0].set_ylabel('UTM Y coordinate (m)')
    ax[1, 0].set_xlim([input_df['x0'].min(), input_df['x0'].max()])

    if np.diff(ax[1, 0].get_ylim()) < 200:
        y_min = min(ax[1, 0].get_ylim())
        y_max = max(ax[1, 0].get_ylim())
        y_avg = np.mean([y_min, y_max])
        ax[1, 0].set_ylim([y_avg-100, y_avg+100])

    ii = 1
    if hist:
        ii += 1
        # Snow depth probability function:
        ax[ii, 0].bar(hs_hist.index, hs_hist['Count'], dhs, edgecolor='steelblue', color='lightsteelblue')
        ax[ii, 0].plot(hs_x, hs_pdf_fit.pdf(hs_x), 'k')
        ax[ii, 0].set_ylabel('PDF')
        ax[ii, 0].set_xlabel('Snow depth (m)')
        ax[ii, 0].set_xlim([hs_hist.index.min(), hs_hist.index.max()])
        ax[ii, 0].text(0.8, 0.9, statstr, bbox=statbox, transform=ax[ii, 0].transAxes,
                fontsize=10, verticalalignment='top')

    if sv_order:
        ii += 1
        ax[ii, 0] = semivariogram(hs_vs, ax=ax[ii, 0])
        ax[ii, 0].set_xlim([0, max(ax[ii, 0].get_xlim())])

    plt.subplots_adjust(top=0.95,
                        left=0.1,
                        right=0.9,
                        bottom=0.05,
                        hspace=0.4)

    return fig


def semivariogram(input_df, n_order=[5, 20, 50]):
    smooth_data, hs_vs = analysis.semivariogram.compute(input_df, n_order)


def smooth_order(input_df, n_orders=None, ax=None):
    """

    :param input_df:
    :param n_orders:
    :return:
    """
    if n_orders is None:
        n_orders = input_df.n_order.unique()
    else:
        pass

    if ax is None:
        fig, ax = plt.subplots()
    else:
        pass
    n_orders = sorted(n_orders)
    alpha_n = np.linspace(0.3, 1, len(n_orders))

    for ii, n_order in enumerate(n_orders):
        st_plot_df = input_df.loc[input_df.n_order == n_order]
        #hs_plot_df = hs_vs.loc[hs_vs.n_order == n_order]
        ax.plot(st_plot_df.index, st_plot_df['SnowDepth'], label='$n_{order}=$%s' % n_order, alpha=alpha_n[ii])
    ax.set_xlabel('Snow depth (m)')
    ax.set_ylabel('Semivariogram (m)')
    ax.legend(fancybox=True, facecolor='lightsteelblue', framealpha=0.5, edgecolor='steelblue')
    return ax

def semivariogram(input_df, n_orders=None, ax=None, scaled=True):
    """

    :param input_df:
    :param n_orders:
    :return:
    """
    if n_orders is None:
        n_orders = sorted(input_df.n_order.unique())
    else:
        pass

    if ax is None:
        fig, ax = plt.subplots()
    else:
        pass
    for n_order in n_orders:
        hs_plot_df = input_df.loc[input_df.n_order == n_order]
        if scaled:
            if n_order == n_orders[0]:
                y = hs_plot_df['Semivariogram']
                y_mean = np.mean(y)
            else:
                y = y_mean * hs_plot_df['Semivariogram']/hs_plot_df['Semivariogram'].mean()
        else:
            y = hs_plot_df['Semivariogram']
        ax.plot(hs_plot_df.index, y, label='$n_{order}=$%d' % n_order)

    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Semivariogram (m)')
    ax.legend(fancybox=True, facecolor='lightsteelblue', framealpha=0.5, edgecolor='steelblue')
    return ax
