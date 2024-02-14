# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:55:46 2023

@author: david

This is an implementation of UNET with several user-specified options.
It draws in part on these UNET implementations (especially the first one):
    https://github.com/VidushiBhatia/U-Net-Implementation/tree/main
    https://keras.io/examples/vision/oxford_pets_image_segmentation/

There is a GroupNormalization layer in Keras, but only in version 2.11 on, and 
to use GPU on Windows, I need to use 2.10 or earlier. So I've used Group Normalization 
from tensorflow_addons instead.
"""

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, SeparableConv2D
from tensorflow.keras.layers import ReLU, ELU
from tensorflow.keras.layers import MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras.layers import concatenate

def BlockConvolutions(conv, n_filters, dropout_prob=0.0, activation='relu', norm='batch', separable=False, dropout_always=False):
    """
    This is the two convolutions part of an encoder or decoder block.
    It is done as a separate function because it is the part that would be repeated in both.
    """
    # Add 2 Conv Layers with a 3x3 kernel size and the specified number of filters.
    #HeNormal initialization: https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal
    #It looks like HeNormal is the same initialization that Ronneberger et al. (2015) used.
    # 'Same' padding will pad the input to conv layer such that the output has the same height and width (hence, is not reduced in size).
    for i in range(2): #Each block involves 2 convolutions.
        if not separable:
            conv = Conv2D(n_filters, 3, activation=None, padding='same', kernel_initializer='HeNormal')(conv)
        else:
            #Separable convolution splits the convolution into 2 parts: one (3x3x1) within each channel and one (1x1xn_channels) across channels.
            #This decreases the total number of parameters, in comparison to the (3x3x3) kernel used in the non-separable version.
            conv = SeparableConv2D(n_filters, 3, activation=None, padding='same', kernel_initializer='HeNormal')(conv)
        # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
        if norm == 'batch':
            conv = BatchNormalization()(conv)
        elif norm == 'group':
            conv = GroupNormalization()(conv)
        elif norm == 'layer':
            conv = LayerNormalization()(conv)
        elif not norm is None:
            print('Error: Unrecognized normalization type.')
        if activation == 'relu':
            conv = ReLU()(conv)
        elif activation == 'elu':
            conv = ELU()(conv)
        else:
            print('Error: Unrecognized Activation Function')
    
    # Monte Carlo dropout can help to reduce over-fitting.
    if dropout_prob > 0:     
        if dropout_always:
            conv = tf.keras.layers.Dropout(dropout_prob)(conv,training=True) #This means dropout will be applied even during evaluation.
        else:
            conv = tf.keras.layers.Dropout(dropout_prob)(conv)
        
    return conv

def Encoder(inputs, n_filters, dropout_prob=0.0, max_pooling=True, activation='relu', norm='batch', separable=False, dropout_always=False):
    """
    This is the encoder block of U-NET, which is repeated multiple times with different filter sizes during the encoding stage.
    """
    
    #Add the two convolutional layers.
    conv = BlockConvolutions(inputs, n_filters, dropout_prob=dropout_prob, activation=activation, 
                             norm=norm, separable=separable, dropout_always=dropout_always)

    # Pooling reduces the size of the image while keeping the number of channels the same
    # Pooling has been kept as optional as the last encoder layer does not use pooling (hence, makes the encoder block flexible to use)
    # Below, Max pooling considers the maximum of the input slice for output computation and uses stride of 2 to traverse across input image
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)    
    else:
        next_layer = conv

    # skip connection (without max pooling) will be input to the decoder layer to prevent information loss during transpose convolutions      
    skip_connection = conv
    
    return next_layer, skip_connection

def Decoder(prev_layer_input, skip_layer_input, n_filters, dropout_prob=0.0, activation='relu', norm='batch', separable=False, dropout_always=False):
    """
    This is the decoder block of U-NET.
    It is similar to the encodder block, but instead of reducing the raster size through max pooling, it increases it through Conv2DTranspose.
    It then adds back in the skip_connection from the corresponding Encoder block, and does two Conv2D filters as in the Encoder.
    """
    # Start with a transpose convolution layer to first increase the size of the image, using a 3x3 kernel size.
    up = Conv2DTranspose(n_filters, (3,3), strides=(2,2), padding='same')(prev_layer_input)

    # Merge the skip connection from previous block to prevent information loss.
    merge = concatenate([up, skip_layer_input], axis=3)
    
    #Add the two convolutional layers.
    conv = BlockConvolutions(merge, n_filters, dropout_prob=dropout_prob, activation=activation, 
                             norm=norm, separable=separable, dropout_always=dropout_always)

    return conv

def UNET(input_size=(128, 128, 3), n_filters_start=32, n_steps=5, n_classes=3, activation='relu', dropout_prob = 0.0, first_dropout_layer = 0,
         norm='batch', separable=False, dropout_always=False):
    """
    Combine both encoder and decoder blocks.
    Return the model as output 
    The number of filters starts with n_filters_start, doubles in each step of the encoder, and halves again in each step of the decoder.
    n_steps gives the number of steps of the encoder block. Max pooling is performed at the end of each of these steps except the last one.
    The decoder block then consists of n_steps-1 steps.
    """
    # Input size represent the size of 1 image (the size used for pre-processing) 
    inputs = Input(input_size)

    #Encoder
    n_filters = n_filters_start
    skips = [None]*n_steps
    convs_en = [None]*n_steps
    for i in range(n_steps):
        if i == 0:
            prev_layer = inputs
        else:
            prev_layer = convs_en[i-1]
        if i < n_steps-1:
            max_pooling = True
        else:
            max_pooling = False
        if i >= first_dropout_layer:
            dropout_prob_layer = dropout_prob
        else:
            dropout_prob_layer = 0
        if i > 0:
            n_filters = n_filters*2
        convs_en[i], skips[i] = Encoder(prev_layer, n_filters, dropout_prob=dropout_prob_layer, max_pooling=max_pooling, activation=activation, 
                               norm=norm, separable=separable, dropout_always=dropout_always)
    
    #Decoder
    convs_de = [None]*(n_steps-1)
    for i in range(n_steps-2,-1,-1):
        if i == n_steps-2:
            prev_layer = convs_en[n_steps-1]
        else:
            prev_layer = convs_de[i+1]
        n_filters = n_filters/2
        if i >= first_dropout_layer:
            dropout_prob_layer = dropout_prob
        else:
            dropout_prob_layer = 0
        convs_de[i] = Decoder(prev_layer, skips[i], n_filters, dropout_prob=dropout_prob_layer, activation=activation, norm=norm, 
                              separable=separable, dropout_always=dropout_always)
    
    #Do a 1x1 Conv layer to get the image to the desired size>
    conv_final = Conv2D(n_classes, 1, activation='softmax', padding='same')(convs_de[0])
    
    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=conv_final)
    
    return model