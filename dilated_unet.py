# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 12:26:19 2021

@author: kaushik.dutta
"""

import tensorflow as tf
import io
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Input, concatenate, Add
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.layers import ReLU, ReLU
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K

K.set_image_data_format('channels_last')
smooth = 1
#initializer = TruncatedNormal(mean=0.0, stddev=0.02)
initializer = 'glorot_normal'

def SSIM(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def M_SSIMLoss(y_true, y_pred):
    #y_true = np.expand_dims(y_true,2)
    return 1 - (tf.image.ssim_multiscale(y_true, y_pred, 1.0))

def mix_loss(y_true, y_pred):
    alpha = 0.90
    beta = 0.1
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    l1_loss = K.mean(K.abs(y_true - y_pred))
    l2_loss = K.mean(tf.keras.losses.mean_squared_error(y_true, y_pred))
    total_loss = alpha*ssim_loss + beta*l1_loss
    return total_loss


def denoisenet(filter_size, pretrained_weights = None):
    input_size = (256,256,1);
    initial_filter_size = int(filter_size)
    inputs = Input(shape = input_size)
    conv1 = Conv2D(initial_filter_size, 3, dilation_rate = 1, padding = 'same', kernel_initializer = initializer)(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    conv1 = Conv2D(initial_filter_size, 3, dilation_rate = 1, padding = 'same', kernel_initializer = initializer)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    
    conv2 = Conv2D(initial_filter_size*2, 3, dilation_rate = 2, padding = 'same', kernel_initializer = initializer)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)
    conv2 = Conv2D(initial_filter_size*2, 3, dilation_rate = 2, padding = 'same', kernel_initializer = initializer)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)
    
    conv3 = Conv2D(initial_filter_size*4, 3, dilation_rate = 4, padding = 'same', kernel_initializer = initializer)(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)
    conv3 = Conv2D(initial_filter_size*4, 3, dilation_rate = 4, padding = 'same', kernel_initializer = initializer)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)
    
    merge1 = concatenate([conv2,conv3], axis = 3)
    deconv1 = Conv2D(initial_filter_size*2, 3, dilation_rate = 2, padding = 'same', kernel_initializer = initializer)(merge1)
    deconv1 = ReLU()(deconv1)
    deconv1 = BatchNormalization()(deconv1)
    deconv1 = Conv2D(initial_filter_size*2, 3, dilation_rate = 2, padding = 'same', kernel_initializer = initializer)(deconv1)
    deconv1 = BatchNormalization()(deconv1)
    deconv1 = ReLU()(deconv1)
    
    merge2 = concatenate([conv1,deconv1], axis = 3)
    deconv2 = Conv2D(initial_filter_size, 3, dilation_rate = 1, padding = 'same', kernel_initializer = initializer)(merge2)
    deconv2 = BatchNormalization()(deconv2)
    deconv2 = ReLU()(deconv2)
    deconv2 = Conv2D(initial_filter_size, 3, dilation_rate = 1, padding = 'same', kernel_initializer = initializer)(deconv2)
    deconv2 = BatchNormalization()(deconv2)
    deconv2 = ReLU()(deconv2)
    
    deconv_final = Conv2D(1,1, activation = None)(deconv2)
    output = Add()([deconv_final ,inputs])
    #output = deconv_final
    
    model = Model(inputs = [inputs], outputs = [output])
    model.compile(optimizer = Adam(learning_rate = 1e-4, amsgrad=True, epsilon = 1e-7), loss = SSIMLoss, metrics = [tf.keras.metrics.MeanAbsoluteError(), SSIM])
    if(pretrained_weights):
    	model.load_weights(pretrained_weights) 
    return model

    
    
    
    
    

