#!/usr/bin/python3
# -*- coding: utf-8 -*-

import logging

import matplotlib.pyplot as plt

import magnaprobe_toolbox as mt

import numpy as np

logger = logging.getLogger(__name__)

# -- USER VARIABLES
# Enable plotting with relative coordinate to the transect origin x0=0, y0=0
plot_origin_flag = True

# Data filename
qc_fp = '/mnt/data/UAF-data/working_a/SALVO/20240417-ICE/magnaprobe/salvo_ice_line_magnaprobe-geo1_20240417.a2.csv'
qc_fp = '/mnt/data/UAF-data/working_a/SALVO/20240417-ICE/magnaprobe/salvo_ice_line_magnaprobe-geo1_20240417.a3.csv'
qc_fp = '/mnt/data/UAF-data/working_a/SALVO/20240418-BEO/magnaprobe/salvo_beo_line_magnaprobe-geodel_20240418.a2.csv'
qc_fp = '/mnt/data/UAF-data/working_a/SALVO/20240418-BEO/magnaprobe/salvo_beo_line_magnaprobe-geo2_20240418.a2.csv'
qc_fp = '/mnt/data/UAF-data/working_a/SALVO/20240419-ARM/magnaprobe/salvo_arm_line_magnaprobe-geodel_20240419.a2.csv'
qc_fp = '/mnt/data/UAF-data/working_a/SALVO/20240420-BEO/magnaprobe/salvo_beo_line_magnaprobe-geodel_20240420.a2.csv'
qc_fp = '/mnt/data/UAF-data/working_a/SALVO/20240421-ICE/magnaprobe/salvo_ice_line_magnaprobe-geodel_20240421.a2.csv'
qc_fp = '/mnt/data/UAF-data/working_a/SALVO/20240421-ICE/magnaprobe/salvo_ice_line_magnaprobe-geodel_20240421.a4.csv'

qc_data = mt.load.qc_data(qc_fp)
# check direction
if qc_data.iloc[-1]['Longitude'] > qc_data.iloc[0]['Longitude']:
    qc_data = qc_data.iloc[::-1]

qc_data = mt.tools.compute_distance(qc_data)
qc_data = mt.tools.all_check(qc_data)

# Insert LineLocation
if 'LineLocation' not in qc_data.columns:
    qc_data['LineLocation'] = [np.nan]*len(qc_data)
else:
    qc_data = qc_data.sort_values(by=['LineLocation'])

if len(qc_data) < 201:
    # Show where
    qc_data.loc[qc_data['LineLocation'] > 1.9, 'Calibration'] = 'MISSING'
    # Generate output_filename
    output_fp = mt.io.output_filename(qc_fp)
    mt.export.data(qc_data, output_fp)
    logger.warning('Data point(s) are missing. Examine exported file %s, and insert nan-filled missing row' %output_fp.split('/')[-1])
    if len(qc_data[qc_data['Quality'] > 6]) > 0:
        print('Cleaning is still needed')
elif len(qc_data) > 201:
    # Generate output_filename
    output_fp = mt.io.output_filename(qc_fp)
    mt.export.data(qc_data, output_fp)
    logger.warning('Supplementary point(s) exists. Examine exported file %s, and insert nan-filled missing row' %output_fp.split('/')[-1])
    if len(qc_data[qc_data['Quality'] > 6]) > 0:
        print('Cleaning is still needed')
else:
    # Populate 'LineLocation' with 0, 1, 2, ... 200 distance array
    qc_data['LineLocation'] = np.arange(0, 201, 1)

    # Generate output_filename
    output_fp = mt.io.output_filename(qc_fp)
    mt.export.data(qc_data, output_fp)

    fig_fp = output_fp.split('.')[0]+'.pdf'
    # Plot basic figures:
    site = output_fp.split('_')[2].upper()
    date = output_fp.split('.')[0][-8:]
    if '_longline' in output_fp:
        name = 'Long transect'
    elif '_line' in output_fp:
        name = '200 m transect'
    elif 'library' in output_fp:
        name = 'Library site'

    fig_title = ' - '.join([site.upper(), name, date])

    # Data status plot
    if qc_data['LineLocation'].isna().all():
        plot_df = qc_data.set_index('TrackDistCum', drop=True)
        fig_title += ' GARBAGE LINE LOCATION'
    else:
        plot_df = qc_data.set_index('LineLocation', drop=True)
    input_df = plot_df
    # Move origin points to x0, y0
    if plot_origin_flag:
        plot_df['x0'] = plot_df['X'] - plot_df.iloc[0]['X']
        plot_df['y0'] = plot_df['Y'] - plot_df.iloc[0]['Y']
    else:
        plot_df['x0'] = plot_df['X']
        plot_df['y0'] = plot_df['Y']

    plt.style.use('ggplot')
    fig = mt.io.plot.summary(plot_df)
    fig.suptitle(fig_title)
    plt.savefig(fig_fp)
    plt.show()
