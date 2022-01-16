import configparser
import logging, sys, os
from pathlib import Path

parent_directory = Path(__file__).absolute().parents[1]

config = configparser.ConfigParser()
config.read(Path(parent_directory, "configurations/properties.ini"))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def initDirectories():
    values = configSectionMap("FOLDERS").values()
    for value in values:
        os.makedirs(value.lower(), exist_ok=True)


def configSectionMap(section):
    global config
    dict1 = {}
    options = config.options(section)
    for option in options:
        try:
            dict1[option] = config.get(section, option)
            if dict1[option] == -1:
                logging.debug("skip: %s" % option)
        except:
            logging.debug("exception on %s!" % option)
            dict1[option] = None
    return dict1
