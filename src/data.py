"""
Copyright (c) 2022 InterDigital R&D France
All rights reserved. 
Licensed under BSD 3 clause clear license - check license.txt for more information
"""


import tensorflow as tf
import numpy as np
from PIL import Image
import pickle 
from numpy import asarray
import os
from math import ceil
from math import log10, sqrt
import random


class Data:
    def __init__(self,list_IDs, list_levels, input_dim = 256, batch_size = 1, task= "synthesis"):
        self.list_IDs=list_IDs
        self.list_levels = list_levels
        self.task = task
        self.input_dim=input_dim
        self.batch_size = batch_size
        self.list_IDs_org, self.list_IDs_fg, self.list_levels = self.load_pkl_multilevel(list_IDs,list_levels)
        self.on_epoch_end()
    
    def load_pkl_multilevel(self, list_IDs, list_levels):
        pickle_in = open(list_IDs,'rb')
        list_IDs_org = pickle.load(pickle_in)["org"]
        pickle_in.close()
        pickle_in = open(list_IDs,'rb')
        list_IDs_fg = pickle.load(pickle_in)["fg"]
        pickle_in.close()        
        pickle_in = open(list_levels,'rb')
        list_levels = pickle.load(pickle_in)
        pickle_in.close()         
        return  list_IDs_org, list_IDs_fg, list_levels

    def batches(self):
        return int(np.floor(len(self.list_IDs_org) / self.batch_size))

    def on_epoch_end(self):
      self.indexes  = np.arange(len(self.list_IDs_org))
      return self.indexes
  
    def data_generation(self):
        X = []
        for i, ID in enumerate(self.list_IDs_temp):
            image =  Image.open(ID)
            image = asarray(image) 
            image = tf.cast(image, tf.float32)
            if  "removal" in self.task:
                image = (image /127.5)-1
            X.append(image)
        return tf.stack(X)


    
    def generate_samples_multilevel(self,index):
        indexes = self.indexes[index * self.batch_size:(index+1) * self.batch_size]
        self.list_IDs_temp = [self.list_IDs_org[k] for k in indexes]
        X_org = self.data_generation()       
        self.list_IDs_temp = [self.list_IDs_fg[k] for k in indexes]
        X_fg = self.data_generation()
        X_levels = np.empty((self.batch_size, self.input_dim, self.input_dim, 1), dtype=np.float32)
        for i, ID in enumerate(self.list_IDs_temp):
            X_levels[i,:,:,:] = np.full((self.input_dim, self.input_dim, 1), self.list_levels[ID])
        if "removal" in self.task:
            return X_fg, X_org, X_levels
        if "synthesis" in self.task:
            return X_org, X_fg, X_levels
    



def process_input(input_image,normalize):
    image =  Image.open(input_image)
    image = asarray(image,dtype=np.float32)
    image = tf.cast(image, tf.float32) 
    image = tf.expand_dims(image, 0, name=None)
    original_shape = image.shape
    width_add = ceil(original_shape[1] / 32 ) * 32 - original_shape[1]
    height_add = ceil(original_shape[2] / 32 ) * 32  - original_shape[2]
    if len(original_shape) != 3 :
        paddings = tf.constant([[0,0],[0, width_add], [0, height_add],[0,0]])
    else:
        paddings = tf.constant([[0,0],[0, width_add], [0, height_add]])
    image = tf.pad(image, paddings, "SYMMETRIC")

    if normalize:
        image = (image /127.5)-1
    return image, original_shape



def pickle_files(path, levels):
    arr = []
    for root, dirs, files in os.walk(path +"fg/01/"):
        for file in files:
            if(".tiff" in file):
                arr.append(os.path.join(root,file))

    partition={}
    org=[]
    fg=[]
    level = {}

    for i in range(len(arr)):
        name = arr[i]
        for j in range(len(levels)): 
            fg.append(name.replace("fg/01/", "fg/"+ str(levels[j]).replace(".", "")+"/"))
            org.append(name.replace("fg/01/", "org/"))
            level[name.replace("fg/01/", "fg/"+ str(levels[j]).replace(".", "")+"/")]= levels[j]
  

    c = list(zip(fg, org))

    random.shuffle(c)

    fg, org= zip(*c)        
    partition["org"] = org
    partition["fg"] = fg
    list_IDs = "data/list_samples.pickle"
    filehandler = open(list_IDs,"wb")
    pickle.dump(partition,filehandler)
    filehandler.close()
 
    list_levels = "data/list_levels.pickle"
    filehandler = open(list_levels,"wb")
    pickle.dump(level,filehandler)
    filehandler.close()
    return list_IDs, list_levels

