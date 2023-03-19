import tensorflow as tf
import io
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Input, Concatenate, Add, GlobalAveragePooling2D, Multiply
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.layers import ReLU, ReLU
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K


def SSIM(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    
def SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

#def learning_rate(epochs)

#eam_filter_size = 64
def eamblock(input, filter_size):
    eam_filter_size = filter_size
    print("EAM Filter Size", eam_filter_size)
    block_1_1 = Conv2D(eam_filter_size, (3,3), dilation_rate = 1, padding = 'same', activation = 'relu')(input)
    block_1_2 = Conv2D(eam_filter_size, (3,3), dilation_rate = 2, padding = 'same', activation = 'relu')(block_1_1)
    block_1_3 = Conv2D(eam_filter_size, (3, 3), dilation_rate = 3, padding='same', activation='relu')(input)
    block_1_4 = Conv2D(eam_filter_size, (3, 3), dilation_rate = 4, padding='same', activation='relu')(block_1_3)
    block_1_concat = Concatenate(axis = -1)([block_1_2, block_1_4])
    block_1 = Conv2D(eam_filter_size, (3, 3), padding = 'same', activation = 'relu')(block_1_concat)
    block_1_add = Add()([block_1, input])

    block_2_1 = Conv2D(eam_filter_size, (3,3), padding = 'same', activation = 'relu')(block_1_add)
    block_2_2 = Conv2D(eam_filter_size, (3,3), padding = 'same', activation = 'relu')(block_2_1)
    block_2_add = Add()([block_2_2, block_1_add])

    block_3_1 = Conv2D(eam_filter_size, (3,3), padding = 'same', activation = 'relu')(block_2_add)
    block_3_2 = Conv2D(eam_filter_size, (3,3), padding = 'same', activation = 'relu')(block_3_1)
    block_3_3 = Conv2D(eam_filter_size, (1,1), padding = 'same')(block_3_2)
    block_3_add = Add()([block_3_3, block_2_add])
    block_3_add = ReLU()(block_3_add)

    pooling_4_1 = GlobalAveragePooling2D()(block_3_add)
    block_4_2 = tf.expand_dims(pooling_4_1, 1)
    block_4_2 = tf.expand_dims(block_4_2, 1)
    block_4_3 = Conv2D(int(eam_filter_size/16), (3,3), padding = 'same', activation = 'relu')(block_4_2)
    block_4_4 = Conv2D(eam_filter_size, (3,3), padding = 'same', activation = 'sigmoid')(block_4_3)
    mult = Multiply()([block_4_4, block_3_add])
    return mult

def ridnet(loss_func, filter_size, pretrained_weights = None):
    inputs = Input(shape = (128,128,1))
    eam_filter_size = filter_size
    feat_ext = Conv2D(eam_filter_size, (3,3), padding = 'same')(inputs)
    eam_block_1 = eamblock(feat_ext, filter_size)
    eam_block_2 = eamblock(eam_block_1, filter_size)
    eam_block_3 = eamblock(eam_block_2, filter_size)
    eam_block_4 = eamblock(eam_block_3, filter_size)
    #add_layer_1 = Add()([feat_ext,eam_block_4])
    feat_ext_1 = Conv2D(1, (3,3), padding='same')(eam_block_4)
    final_layer = Add()([feat_ext_1, inputs])
    model = Model(inputs = [inputs], outputs = [final_layer])
    model.compile(optimizer = Adam(learning_rate = 1e-4, amsgrad=True, epsilon = 1e-7), loss = SSIMLoss, metrics = [tf.keras.metrics.MeanAbsoluteError(), SSIM])
    if(pretrained_weights):
    	model.load_weights(pretrained_weights) 
    return model

















