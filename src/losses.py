"""
Copyright (c) 2022 InterDigital R&D France
All rights reserved. 
Licensed under BSD 3 clause clear license - check license.txt for more information
"""
import tensorflow as tf
import numpy as np


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)


def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  total_gen_loss = gan_loss + (0.1 * l1_loss)
  return total_gen_loss, gan_loss, l1_loss



def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
  total_disc_loss = ( real_loss + generated_loss)
  return total_disc_loss


     
class LOSS(object):
    def __init__(self, k1=0.01, k2=0.02, L=2, window_size=11, channel=3):
        self.k1 = k1
        self.k2 = k2           # constants for stable
        self.L = L             # the value range of input image pixels
        self.WS = window_size
        self.channel= channel

    def _tf_fspecial_gauss(self, size, sigma=1.5):
        """Function to mimic the 'fspecial' gaussian MATLAB function"""
        x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

        x_data = np.expand_dims(x_data, axis=-1)
        x_data = np.expand_dims(x_data, axis=-1)

        y_data = np.expand_dims(y_data, axis=-1)
        y_data = np.expand_dims(y_data, axis=-1)

        x = tf.constant(x_data, dtype=tf.float32)
        y = tf.constant(y_data, dtype=tf.float32)

        g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g / tf.reduce_sum(g)
    
    

    def ssim_loss(self, img1, img2):
        """
        The function is to calculate the ssim score
        """
        window = self._tf_fspecial_gauss(size=self.WS)  # output size is (window_size, window_size, 1, 1)


        (_, _, _, self.channel) = img1.shape.as_list()

        window = tf.tile(window, [1, 1, self.channel, 1])

        mu1 = tf.nn.depthwise_conv2d(img1, window, strides = [1, 1, 1, 1], padding = 'VALID')
        mu2 = tf.nn.depthwise_conv2d(img2, window, strides = [1, 1, 1, 1], padding = 'VALID')

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        img1_2 = img1*img1
        sigma1_sq = tf.subtract(tf.nn.depthwise_conv2d(img1_2, window, strides = [1 ,1, 1, 1], padding = 'VALID') , mu1_sq)
        img2_2 = img2*img2
        sigma2_sq = tf.subtract(tf.nn.depthwise_conv2d(img2_2, window, strides = [1, 1, 1, 1], padding = 'VALID') ,mu2_sq)
        img12_2 = img1*img2
        sigma1_2 = tf.subtract(tf.nn.depthwise_conv2d(img12_2, window, strides = [1, 1, 1, 1], padding = 'VALID') , mu1_mu2)

        c1 = (self.k1*self.L)**2
        c2 = (self.k2*self.L)**2
        
        v1 = 2.0 * sigma1_2 +c2
        v2 = sigma1_sq + sigma2_sq +c2
        # print(v2)
        cs =  v1 / v2 
        ssim_map = ((2*mu1_mu2 + c1)* v1) / ((mu1_sq + mu2_sq + c1)* v2)

        return tf.reduce_mean(ssim_map),  tf.reduce_mean(cs)

  
    def ms_ssim_l1(self, img1, img2,level=5):
      weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
      mssim = []
      mcs = []
      
      img1_zero = tf.pad(img1, [[0,0], [5, 5], [5,5], [0,0]], "CONSTANT")
      img2_zero = tf.pad(img2, [[0,0], [5, 5], [5, 5], [0,0]], "CONSTANT")
      
      l1 = tf.abs(img1_zero - img2_zero)

      #Padding reflect for ssim computation
      img1 = tf.pad(img1, [[0,0], [5, 5], [5,5], [0,0]], "REFLECT")
      img2 = tf.pad(img2, [[0,0], [5, 5], [5, 5], [0,0]], "REFLECT")
      
      for l in range(level):
        ssim_map, cs_map = self.ssim_loss(img1, img2)
        mssim.append(ssim_map)
        mcs.append(cs_map)

        filtered_im1 = tf.nn.avg_pool2d(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool2d(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2       
              
      mssim = tf.stack(mssim)
      mcs = tf.stack(mcs)
      
      mssim = (mssim + 1) / 2
      mcs = (mcs + 1) / 2

      pow1 = mcs ** weight
      pow2 = mssim ** weight

      ms_loss  = 1 - (tf.math.reduce_prod(pow1[:-1]) * pow2[-1])
      window = self._tf_fspecial_gauss(size=self.WS)  # output size is (window_size, window_size, 1, 1)
      window = tf.tile(window, [1, 1, self.channel, 1])
      weighted_l1 = tf.nn.depthwise_conv2d(l1, window, strides = [1, 1, 1, 1], padding = 'VALID')
      weighted_l1 = tf.reduce_mean(weighted_l1)
      
      return 0.84 * ms_loss + 0.16 * weighted_l1


    def l1(self, img1, img2):
        return tf.reduce_mean(tf.abs(img1 - img2))
