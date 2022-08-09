"""
Copyright (c) 2022 InterDigital R&D France
All rights reserved. 
Licensed under BSD 3 clause clear license - check license.txt for more information
"""


import numpy as np
import imageio
from matplotlib import pyplot as plt

def visualize_fg(input_image, gen_output, target_image, X_level, name):               
  prediction =  np.clip(gen_output[0],0.0,255.0)
  prediction = prediction.astype(np.uint8)

  target_input = target_image[0]
  target_input =  np.clip(target_input,0.0,255.0)
  target_input = target_input.astype(np.uint8)
  
  test_input = input_image[0]
  test_input =  np.clip(test_input,0.0,255.0)
  test_input = test_input.astype(np.uint8)
  
  level = X_level[0,0,0]
  level = level[0]
  plt.figure(figsize=(15, 15))

  display_list = [test_input, prediction, target_input]
  title = ['Input' , 'Predicted', 'Target (level = '+ str(level)+')']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    plt.imshow(display_list[i] )
    plt.axis('off')
  plt.savefig("train_output/test_" + name+ ".tiff")
  plt.close()


def visualize_org(input_image, gen_output, target_image, X_level, name):               
  prediction = gen_output[0]*0.5 + 0.5
  prediction =  np.clip(prediction,0.0,1.0)*255
  prediction = prediction.astype(np.uint8)
  
  target_input = target_image[0]*0.5 +0.5
  target_input =  np.clip(target_input,0.0,1.0)*255
  target_input = target_input.astype(np.uint8)
  

  test_input = input_image[0]*0.5 +0.5
  test_input =  np.clip(test_input,0.0,1.0)*255
  test_input = test_input.astype(np.uint8)

  level = X_level[0,0,0]
  level = level[0]
  plt.figure(figsize=(15, 15))

  display_list = [test_input, prediction, target_input]
  title = ['Input (level = '+ str(level)+')' , 'Predicted', 'Target']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    plt.imshow(display_list[i] )
    plt.axis('off')
  plt.savefig("train_output/test_" + name+ ".tiff")
  plt.close()

   
