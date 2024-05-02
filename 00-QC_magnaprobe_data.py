import magnaprobe_toolbox as mt

raw_fp = '/mnt/data/UAF-data/raw/SALVO/20240416-ARM/magnaprobe/salvo_arm_longline_magnaprobe-geo1_20240416.00.dat'
raw_fp = '/mnt/data/UAF-data/raw/SALVO/20240417-ICE/magnaprobe/salvo_ice_line_magnaprobe-geo1_20240417.00.dat'
# -- USER VARIABLES
# Enable plotting with relative coordinate to the transect origin x0=0, y0=0
# ESPG code for local projection
local_ESPG = '3338'  # for Alaska

# Read raw magnaprobe data
raw_df = mt.load.raw_data(raw_fp, local_EPSG=local_ESPG)

# Perform quality and calibration check
raw_df = mt.tools.all_check(raw_df, display=False)

# Define output filename
output_fp = mt.load.output_filename(raw_fp)

# Export to filename for manual analysis
mt.export.data(raw_df, output_fp)
