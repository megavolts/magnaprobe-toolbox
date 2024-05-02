#!/usr/bin/python3
# -*- coding: utf-8 -*-

import logging

import matplotlib.pyplot as plt

import magnaprobe_toolbox as mt

logger = logging.getLogger(__name__)

# -- USER VARIABLES
# Enable plotting with relative coordinate to the transect origin x0=0, y0=0
plot_origin_flag = True

# Data filename
qc_fp = '/mnt/data/UAF-data/working_a/SALVO/20240416-ARM/magnaprobe/salvo_arm_longline_magnaprobe-geo1_20240416.a2.csv'

qc_data = mt.load.qc_data(qc_fp)
qc_data = mt.tools.compute_distance(qc_data)
qc_data = mt.tools.all_check(qc_data)

# Since the longline transect is measured without a fixed distance, we don't check for distance (flag = 8)
qc_data[qc_data['Quality'] == 8, 'Quality'] = 0
if len(qc_data[qc_data['Quality'] > 6]) > 0:
    print('Cleaning is still needed')

# Generate output_filename
output_fp = mt.io.output_filename(qc_fp)
mt.export.data(qc_data, output_fp)

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
plot_df = qc_data.set_index('TrackDistCum', drop=True)
input_df = plot_df
# Move origin points to x0, y0
if plot_origin_flag:
    plot_df['x0'] = plot_df['X'] - plot_df.iloc[0]['X']
    plot_df['y0'] = plot_df['Y'] - plot_df.iloc[0]['Y']
else:
    plot_df['x0'] = plot_df['X']
    plot_df['y0'] = plot_df['Y']

plt.style.use('ggplot')

# Define figure subplots
fig = mt.io.plot.summary(plot_df)
fig.suptitle((' - '.join([site.upper(), name, date])))

plt.show()
