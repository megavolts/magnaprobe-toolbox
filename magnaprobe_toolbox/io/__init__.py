# List of headers to strip

__all__ = ['lower_cal', 'upper_cal', 'col2remove_l', 'header_order', 'output_filename']

import os
import logging

logger = logging.getLogger(__name__)

col2remove_l = ['latitude_a', 'latitude_b', 'Longitude_a', 'Longitude_b', 'fix_quality',
               'nmbr_satellites', 'HDOP', 'altitudeB', 'DepthVolts', 'LatitudeDDDDD', 'LongitudeDDDDD', 'month',
               'dayofmonth', 'hourofday', 'minutes', 'seconds', 'microseconds', 'depthcm']

header_order = ['Record', 'Counter', 'Timestamp', 'SnowDepth', 'Latitude', 'Longitude', 'X', 'Y', 'Z', 'TrackDist',
                'TrackDistCum', 'DistOrigin', 'LineLocation', 'Quality', 'Calibration', 'Battvolts']


lower_cal = 0.02  # m
upper_cal = 1.18  # m

def output_filename(in_fp):
    """
    Generate output filepath base on the input raw filepath. If the file directory does not exist it is created.
    :param in_fp: str
    :return: str
        A string containing the filename in which the output data is writtent
    """
    base_dir = os.path.dirname(in_fp)
    out_dir = base_dir.replace('/raw/', '/working_a/')

    # Input filename
    input_fn = os.path.basename(in_fp)
    if '.00.' in input_fn:
        output_fn = input_fn.replace('.00.', '.a1.')
    elif '.a' in input_fn:
        _increment_number = int(input_fn.split('.')[-2][1:])
        _next_number = _increment_number + 1
        if _next_number == 0:
            # According to ARM guideline a0 is only for raw data exported to NetCDF
            # https://www.arm.gov/guidance/datause/formatting-and-file-naming-protocols
            _next_number = 1
        output_fn = input_fn.replace(str('a%.0f' % _increment_number), str('a%.0f' % _next_number))
    out_file = os.path.join(out_dir, output_fn)
    out_dir = ('/').join(out_file.split('/')[:-1])
    if not os.path.exists(out_dir):
        logger.info(str('Creating output file directory: %s' %out_dir))
        os.makedirs(('/').join(out_file.split('/')[:-1]))
    return out_file
