__all__ = ['load']

import logging
from os.path import exists, basename
import yaml

# Logger:
logger = logging.getLogger(__name__)


def load(input_fp):
    """
    Load the `*.00.yaml` config file associated to the data file, if it exists within the same directory
    :param input_fp: string
        Filepath to the data file
    :return config: dictionary
        Dict containing data processing
    """
    config_fp = input_fp.split('.')[0] + '.00.yaml'
    if exists(config_fp):
        with open(config_fp, 'r') as yaml_fp:
            config = yaml.safe_load(yaml_fp)
    else:
        logger.warning('No config file associated to %s exist. Creating empty config dictionnary' % basename(input_fp))
        config = {}

    return config


def save(config, input_fp, mode='w', display=True):
    """
    Save config to a yaml file within the same directory as the input file. The filename is identical to the input
    filename with the extension '.00.yaml'.

    :param input_fp:
    :param write_type: 'w', 'a'
        'o': to overwrite file
        'a': to append to file
    :return str:
    """

    config_fp = input_fp.split('.')[0] + '.00.yaml'
    with open(config_fp, mode) as fp:
        yaml.dump(config, fp)
    if display:
        print('Conf exported to %s' % config_fp)
    return config_fp
