import os
import ast
from configparser import ConfigParser

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')


class Config:
    """
    This class read a config file and acts as a configiguration object.

    Input: config path

    Each section in config file will become an attribute named in lower case,
    which will be bound to a dict containing values in the section. Although the
    default behavior of ConfigParser will read all values in string format, we
    parse this value so that boolean values and numeric values can be in
    currect format.

    Values in 'DEFAULT' section will directly becomes an attribute in lower cases.

    Note that sections and attributes will always in lower case.

    >>> config = Config('./main.config.ini')
    >>> config.datadir
    'data'
    >>> config.generator
    {
        # a dictionary
    }
    """
    def __init__(self, path):
        config = ConfigParser()
        config.read(path)

        get_abs_path = lambda p: os.path.join(PROJECT_DIR,
                                              config['DEFAULT']['DataDir'],
                                              p)

        # Default value becomes attribute
        for key, value in config['DEFAULT'].items():
            setattr(self, key.lower(), Config.get_value(value))

        self.sections = [s for s in config.sections() if s != 'DATA']

        for section in self.sections:
            sec = {key: Config.get_value(value)
                   for key, value in config[section].items()
                   if key not in config['DEFAULT']}

            sec_keys = list(sec.keys())
            for sec_name in sec_keys:
                if sec_name.startswith('data_'):
                    sec[sec_name[5:]] = get_abs_path(sec.pop(sec_name))

            setattr(self, section.lower(), sec)


    @staticmethod
    def get_value(v):
        if v == 'False' or v == 'false':
            return False
        elif v == 'True' or v == 'True':
            return True
        else:
            try:
                x = float(v)
                if x == int(x):
                    return int(x)
                return x
            except ValueError:
                try:
                    x = ast.literal_eval(v)
                    return x
                except ValueError:
                    return v

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.sum = 0
        self.count = 0

    def reset(self):
        self.sum = 0
        self.count = 0

    @property
    def avg(self):
        return 0 if self.count == 0 else self.sum / self.count

    def update(self, val, weight=1):
        self.sum += val * weight
        self.count += weight
