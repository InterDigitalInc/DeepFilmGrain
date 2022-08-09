"""
Copyright (c) 2022 InterDigital R&D France
All rights reserved. 
Licensed under BSD 3 clause clear license - check license.txt for more information
"""

from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2DTranspose, ZeroPadding2D, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import ReLU, LeakyReLU
from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.layers import BatchNormalization,Activation, Add


###### Define encoder block ########### CONV + BN + ReLu
def encoder_block(inputs, num_filters, batchnorm= True, padding="same", strides =2):
  init = RandomNormal(stddev=0.02)
  x = Conv2D(num_filters, kernel_size=(4,4), strides=(2,2), padding="same", kernel_initializer=init,use_bias=False)(inputs)
  if batchnorm:
    x = BatchNormalization()(x)
  x =ReLU()(x)
  return x

###### Define decoder block ########### CONV + BN + ReLu + Concat
def decoder_block(inputs, skip_in, num_filters, batchnorm=True):
  init = RandomNormal(stddev=0.02)
  x = Conv2DTranspose(num_filters, kernel_size=(4, 4), strides=(2,2), padding="same", kernel_initializer=init,use_bias=False)(inputs)
  if batchnorm:
    x = BatchNormalization()(x)
  x = ReLU()(x)
  x = Concatenate()([x, skip_in])
  return x

###### Define encoder block for disc ########### CONV + BN + LeakyReLu
def encoder_block_disc(inputs, num_filters, batchnorm= True, padding="same", strides =2):
  init = RandomNormal(stddev=0.02)
  x = Conv2D(num_filters, kernel_size=(4,4), strides = (2,2), padding="same", kernel_initializer=init,use_bias=False)(inputs)
  if batchnorm:
    x = BatchNormalization()(x)
  x =LeakyReLU()(x)
  return x


####### Define generator based on unet #############
def unet(input_dim=None, output_channel=3, normalized = False):
    init = RandomNormal(stddev=0.02) 
    inputs = Input((input_dim, input_dim, output_channel))
    level = Input((input_dim, input_dim, 1))
    x = Concatenate()([inputs,level])
    #######  5 encoder  blocks #########
    s1 = encoder_block(x, 64, batchnorm= False)
    s2 = encoder_block(s1, 128)
    s3 = encoder_block(s2, 256)
    s4= encoder_block(s3, 512)
    
    b = encoder_block(s4, 512 )
    
 
    d4 = decoder_block(b, s4, 512)
    d5 = decoder_block(d4, s3, 256)
    d6 = decoder_block(d5, s2, 128)
    d7 = decoder_block(d6, s1, 64)
    
    ###### Output ########
    outputs = Conv2DTranspose(output_channel, kernel_size=(4, 4), strides=(2,2), padding="same", kernel_initializer=init)(d7)
    if normalized == True:
        outputs = Activation('tanh')(outputs)
    model = Model([inputs, level], outputs, name="U-Net")
    return model

def patchgan(input_dim=256, output_channel=3):
    init = RandomNormal(stddev=0.02) 
    real = Input((input_dim, input_dim, output_channel))
    fake = Input((input_dim, input_dim, output_channel))
    level = Input((input_dim, input_dim, 1))

    inputs = Concatenate()([real,fake,level])
    
    a1 = encoder_block_disc(inputs, 64, batchnorm= False)
    a2 = encoder_block_disc(a1, 128)
    a3 = encoder_block_disc(a2, 256)  
    
    a = ZeroPadding2D()(a3)  

    x = Conv2D(512, kernel_size=(4,4), strides = (1,1), kernel_initializer=init)(a)
    x = BatchNormalization()(x)
    x = ReLU()(x)
     
    a = ZeroPadding2D()(x)
    
    output = Conv2D(output_channel, kernel_size=(4,4), strides =(1,1), kernel_initializer=init, activation ="sigmoid")(a)
    
    model = Model([real, fake,level],output, name="PatchGAN")
    return model


####### Define generator based on unet with res blocks #############
def res_unet(filter_root, depth, input_dim=None, output_channel=3, normalized= False, blind=False):
    
    inputs = Input((input_dim, input_dim, output_channel))
    level = Input((input_dim, input_dim, 1))
    x = Concatenate()([inputs,level])  
    
    if blind == True:  
        x = inputs
               
        
    # Dictionary for long connections
    long_connection_store = {}
    # Down sampling
    for i in range(depth):
        out_channel = 2**i * filter_root
        
        # Residual/Skip connection
        res = Conv2D(out_channel, kernel_size=1, padding='same', use_bias=False)(x)   
        
        # First Conv Block with Conv, BN and activation
        conv1 = Conv2D(out_channel, kernel_size=3, padding='same')(x)
        conv1 = BatchNormalization()(conv1)
        act1 = Activation('relu')(conv1)
        
        # Second Conv block with Conv and BN only
        conv2 = Conv2D(out_channel, kernel_size=3, padding='same')(act1)
        conv2 = BatchNormalization()(conv2)
        
        # Add + ReLU
        resconnection = Add()([res, conv2])
        act2 = Activation('relu')(resconnection)
        
        # Max pooling
        if i < depth - 1:
            long_connection_store[str(i)] = act2
            x = MaxPooling2D(padding='same')(act2)
        else:
            x = act2
    # Upsampling
    for i in range(depth - 2, -1, -1):
        out_channel = 2**(i) * filter_root
        long_connection = long_connection_store[str(i)]
        
        # Upsampling + conv
        up1 = UpSampling2D()(x)
        up_conv1 = Conv2D(out_channel, 2, activation='relu', padding='same')(up1)   
        
        #Long skip connection
        up_conc = Concatenate(axis=-1)([up_conv1, long_connection]) 
        
        # Residual/Skip connection
        res = Conv2D(out_channel, kernel_size=1, padding='same', use_bias=False)(up_conc)
        
        # First Conv Block with Conv, BN and activation
        up_conv2 = Conv2D(out_channel, 3, padding='same')(up_conc)
        up_conv2 = BatchNormalization()(up_conv2)
        up_act1 = Activation('relu')(up_conv2)
        
        # Second Conv block with Conv and BN only
        up_conv2 = Conv2D(out_channel, 3, padding='same')(up_act1)
        up_conv2 = BatchNormalization()(up_conv2)

        # Add + ReLU
        resconnection = Add()([res, up_conv2])
        x = Activation('relu')(resconnection)
        
        
    # Final convolution
    output = Conv2D(output_channel, 1, padding='same',name='output')(x)
    if normalized == True:
        output = Activation('tanh')(output)
    if blind == True:
            return Model(inputs, outputs=output, name='Res-UNet-Blind')
    return Model([inputs,level], outputs=output, name='Res-UNet')

