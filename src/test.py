"""
Copyright (c) 2022 InterDigital R&D France
All rights reserved. 
Licensed under BSD 3 clause clear license - check license.txt for more information
"""

import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from numpy import asarray
import imageio 
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from data import process_input


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str, help='path/of/pretrained/model')
    parser.add_argument('--level', type=float, default='0.01', help='0.01, 0.025, 0.05, 0.075, 0.1')
    parser.add_argument('--input_path', type=str, help='path/to/test/set')
    parser.add_argument('--output_path', type=str, help='path/to/test/result')    

    args = parser.parse_args()

    list_levels= [0.01, 0.025, 0.05, 0.075, 0.1]
    
    if ".h5" not in args.pretrained_model:
        raise argparse.ArgumentTypeError(
            'Incorrect pretrained model fromat')

    if args.level not in list_levels:
        raise argparse.ArgumentTypeError(
            'Specified level not supported, supported levels = {0.01, 0.025, 0.05, 0.075, 0.1}')

    if "removal" in args.pretrained_model:
        print("Task : Film grain removal")
    else:
        print("Task : Film grain synthesis")

    print("Model : " + args.pretrained_model)
    level = str(args.level).replace(".", "")
    print("Level : " + level)

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------
    if "removal" in args.pretrained_model:
        L_path = args.input_path 
    else:
        L_path = args.input_path

    E_path = args.output_path 
 
    if not os.path.exists(E_path):
        os.makedirs(E_path)
    paths = os.listdir(L_path)

    ## ----------------------------------------
    ## load model
    ## ----------------------------------------
    model = load_model(args.pretrained_model, compile=False)

    ## ----------------------------------------
    ## Test model
    ## ----------------------------------------
    if "removal" in args.pretrained_model:
        for idx, img in enumerate(paths):
            image_name = L_path + img
            image_L, original_shape = process_input(image_name, True)
            if 'non_blind' in args.pretrained_model:
                shape = image_L.shape
                X_levels = np.empty((1, shape[1], shape[2], 1), dtype=np.float32)
                X_levels[0,:,:,:] = np.full((shape[1], shape[2], 1), args.level)
                image_E = model([image_L, X_levels], training = False)
            else:
                image_E = model(image_L, training = False)

            image_E = image_E[0, :original_shape[1], :original_shape[2],:]     
            image_E = np.array(image_E)*0.5 +0.5
            image_E =  np.clip(image_E,0.0,1.0)*255
            image_E = image_E.astype(np.uint8)
            imageio.imsave(E_path + img, image_E) 

    if "synthesis" in args.pretrained_model:
        for idx, img in enumerate(paths):
            image_name = L_path + img
            image_L, original_shape = process_input(image_name, False)
            shape = image_L.shape
            X_levels = np.empty((1, shape[1], shape[2], 1), dtype=np.float32)
            X_levels[0,:,:,:] = np.full((shape[1], shape[2], 1),  args.level)
            image_E = model([image_L, X_levels], training = True)
            image_E = image_E[0, :original_shape[1], :original_shape[2],:]
            image_E =  np.clip(image_E,0.0,255.0)
            image_E = image_E.astype(np.uint8)
            imageio.imsave(E_path + img, image_E) 


if __name__ == '__main__':
    main()

