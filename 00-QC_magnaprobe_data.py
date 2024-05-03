    import magnaprobe_toolbox as mt
    import shutil
    from os import path

    raw_fp = '/mnt/data/UAF-data/raw/SALVO/20240416-ARM/magnaprobe/salvo_arm_longline_magnaprobe-geo1_20240416.00.dat'
    raw_fp = '/mnt/data/UAF-data/raw/SALVO/20240417-ICE/magnaprobe/salvo_ice_line_magnaprobe-geo1_20240417.00.dat'
    raw_fp = '/mnt/data/UAF-data/raw/SALVO/20240418-BEO/magnaprobe/salvo_beo_line_magnaprobe-geodel_20240418.00.dat'
    raw_fp = '/mnt/data/UAF-data/working_a/SALVO/20240418-BEO/magnaprobe/salvo_beo_longline_magnaprobe-geo2_20240418.00.dat'
    raw_fp = '/mnt/data/UAF-data/working_a/SALVO/20240418-BEO/magnaprobe/salvo_beo_line_magnaprobe-geo2_20240418.00.dat'
    raw_fp = '/mnt/data/UAF-data/raw/SALVO/20240419-ARM/magnaprobe/salvo_arm_line_magnaprobe-geodel_20240419.00.dat'
    raw_fp = '/mnt/data/UAF-data/working_a/SALVO/20240420-BEO/magnaprobe/salvo_beo_line_magnaprobe-geodel_20240420.00.dat'
    raw_fp = '/mnt/data/UAF-data/working_a/SALVO/20240420-BEO/magnaprobe/salvo_beo_longline_magnaprobe-geodel_20240420.00.dat'
    raw_fp = '/mnt/data/UAF-data/working_a/SALVO/20240421-ICE/magnaprobe/salvo_ice_line_magnaprobe-geodel_20240421.00.dat'
    raw_fp = '/mnt/data/UAF-data/working_a/SALVO/20240421-ICE/magnaprobe/salvo_ice_library_magnaprobe-geodel_20240421.00.dat'

    # -- USER VARIABLES
    # Enable plotting with relative coordinate to the transect origin x0=0, y0=0
    # ESPG code for local projection
    local_ESPG = '3338'  # for Alaska

    # Define output filename
    output_fp = mt.load.output_filename(raw_fp)

    # Read raw magnaprobe data
    raw_df = mt.load.raw_data(raw_fp, local_EPSG=local_ESPG)

    # Perform quality and calibration check
    raw_df = mt.tools.all_check(raw_df, display=False)

    # Export to filename for manual analysis
    mt.export.data(raw_df, output_fp)
