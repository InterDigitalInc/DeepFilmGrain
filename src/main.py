"""
Copyright (c) 2022 InterDigital R&D France
All rights reserved. 
Licensed under BSD 3 clause clear license - check license.txt for more information
"""

import tensorflow as tf
import argparse
from train import trainloop
import json


#============================ Arguments parsing ============================ #
parser = argparse.ArgumentParser()
parser.add_argument('--task', default='removal_case_1' , type=str, help='Specify task')
parser.add_argument('--path', type=str, help='Dataset path')
args = parser.parse_args() 
with open('src/config.json', 'r') as config_file:
    data_config = json.load(config_file)

trainloop(data_config, args.task, args.path)

           