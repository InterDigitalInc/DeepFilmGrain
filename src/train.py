"""
Copyright (c) 2022 InterDigital R&D France
All rights reserved. 
Licensed under BSD 3 clause clear license - check license.txt for more information
"""

import logging
import os 


logging.getLogger('tensorflow').setLevel(logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from data import Data, pickle_files
from model import *
from losses import *
from utils import *
import tensorflow as tf
import json
import sys




def trainloop(data_config, task, path):
    if not os.path.exists("train_output"):
        os.makedirs("train_output")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[ logging.FileHandler("debug.log"), logging.StreamHandler(sys.stdout)])
    
    if "gray" in task:
        output_channel=1
        path_channel = "Y"
    else: 
        output_channel=3 
        path_channel = "RGB"

    logging.info('Data loading')

    list_IDs,list_levels =  pickle_files(path, data_config['levels'])

    DATA = Data(list_IDs, list_levels, data_config['input_dim'],  data_config['batch_size'], task)
    list_IDs_org, list_IDs_fg, list_levels = DATA.load_pkl_multilevel(list_IDs, list_levels)
    num_batches = DATA.batches() 
    print(num_batches)
    epochs = data_config["epochs"]
    input_dim = data_config["input_dim"]
    batch_size = data_config["batch_size"]
    generator_optimizer = tf.keras.optimizers.Adam(data_config['learning_rate_alpha_gen'], data_config['learning_rate_beta_gen'] )
    discriminator_optimizer = tf.keras.optimizers.Adam(data_config['learning_rate_alpha_dis'], data_config['learning_rate_beta_gen'] )
    loss = LOSS(k1=0.01, k2=0.03, L=2, window_size=11)
    


    logging.info('Model building')
    GAN = False

    if "synthesis" in task:
        gen_loss = generator_loss
        disc_loss = discriminator_loss
        normalized = False
        
        if "case_1" in task: # U-Net optimized with l1
            generator = unet(input_dim, output_channel, normalized)
            gen_loss = loss.l1
            
        elif "case_2" in task: # cGAN with U-Net as backbone for generator
            generator = unet(input_dim, output_channel, normalized)
            discriminator = patchgan(input_dim, output_channel)
            GAN = True
            
        elif "case_3" in task: # cGAN with U-Net + residual blocks as backbone for generator
            generator = res_unet(64, 5, input_dim, output_channel, normalized, blind=False)    
            discriminator = patchgan(input_dim, output_channel)
            GAN = True
            
        elif  "case_3_gray" in task: # cGAN with U-Net + residual blocks as backbone for generator for grayscale images
            generator = res_unet(64, 5, input_dim, output_channel, normalized, blind=False)    
            discriminator = patchgan(input_dim, output_channel)
            GAN = True
        generator.summary()
            
            
    elif "removal" in task:
        normalized = True
        blind=False
        if "case_1" in task: # U-Net optimized with l1
            generator = unet(input_dim, output_channel, normalized)
            gen_loss = loss.l1
            
        elif "case_2" in task: # U-Net with residual blocks optimized with l1
            generator = res_unet(64, 5, input_dim, output_channel, normalized, blind)
            gen_loss = loss.l1
            
        elif "case_3" in task:  # U-Net with residual blocks optimized with mix of l1 and MS-SSIM
            generator = res_unet(64, 5, input_dim, output_channel, normalized, blind)
            gen_loss = loss.ms_ssim_l1
            
        elif "case_4" in task: # U-Net with residual blocks optimized with mix of l1 and MS-SSIM but blind 
            blind =True
            generator = res_unet(64, 5, input_dim, output_channel, normalized, blind)
            gen_loss = loss.ms_ssim_l1
            
        elif "case_3_gray" in task: # U-Net with residual blocks optimized with mix of l1 and MS-SSIM for grayscale images
            generator = res_unet(64, 5, input_dim, output_channel, normalized, blind)
            gen_loss = loss.ms_ssim_l1
            
        elif "case_4_gray" in task: # U-Net with residual blocks optimized with mix of l1 and MS-SSIM but blind for grayscale images
            blind =True
            generator = res_unet(64, 5, input_dim, output_channel, normalized, blind)
            gen_loss = loss.ms_ssim_l1
        generator.summary()

    

    
    for epoch in range(epochs):
        print("Epoch : " +str(epoch))
        for batch_num in range(num_batches):
            input_image, target_image, X_level = DATA.generate_samples_multilevel(batch_num)
            if "gray" in task:
                input_image= tf.reshape(input_image, (batch_size,256,256,1))
                target_image= tf.reshape(target_image, (batch_size,256,256,1))
                X_level =  tf.reshape(X_level, (batch_size,256,256,1))
                
            if GAN:
                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    #  Forward
                    gen_output = generator([input_image, X_level], training=True)
                    disc_real_output = discriminator([input_image, target_image, X_level], training=True)
                    disc_generated_output = discriminator([input_image, gen_output,X_level], training=True)
                    
                    #  Loss computing
                    gen_total_loss, gen_gan_loss, gen_l1_loss = gen_loss(disc_generated_output, gen_output, target_image)
                    disc_total_loss = disc_loss(disc_real_output, disc_generated_output)

                #  Gradients computing
                generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
                discriminator_gradients = disc_tape.gradient(disc_total_loss, discriminator.trainable_variables)

                #  Gradients application
                generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
                discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
                
            else:
                with tf.GradientTape() as gen_tape: 
                    if "case_4" in task:
                        gen_output = generator(input_image, training=True)
                    else:
                        gen_output = generator([input_image, X_level], training=True)

                    gen_total_loss = gen_loss(gen_output, target_image)

                generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
                generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
            if(batch_num % 20 == 0):                
                if "removal" in task:
                    visualize_org(input_image, gen_output, target_image, X_level, task)                
                elif "synthesis" in task:
                    visualize_fg(input_image, gen_output, target_image, X_level, task)                
                generator.save("train_output/"+task+"_"+str(epoch)+".h5")  


