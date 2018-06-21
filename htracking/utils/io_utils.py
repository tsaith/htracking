import numpy as np
import pandas as pd
import yaml
import os


def read_config(filepath):
    """
    Read the configuration file of YAML format.
    """

    with open(filepath, 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as err:
            print(err)

    return config
