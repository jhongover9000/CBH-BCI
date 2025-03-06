import numpy as np
import mat73
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Activation, Permute, Dropout, Conv2D, MaxPooling2D, 
    AveragePooling2D, Add, SeparableConv2D, DepthwiseConv2D, 
    BatchNormalization, SpatialDropout2D, Reshape, Input, 
    Flatten, MultiHeadAttention, LayerNormalization, Conv1D, 
    Concatenate, Lambda, GlobalAveragePooling2D, multiply
)
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical

from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler, scale
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import shap
import matplotlib.pyplot as plt
from datetime import datetime
import gc
import keras
def proposed(n_timesteps, n_features, n_outputs):
    
    input_1 = Input(shape=(1, n_features, n_timesteps))  # TensorShape([None, 1, 22, 1125])

    block1       = Conv2D(filters=8, kernel_size=(1, 32), use_bias = False, padding='same', data_format="channels_first")(input_1)
    block1       = LayerNormalization()(block1)
    block1       = Activation(activation='elu')(block1)
    
  
    block2       = Conv2D(filters=8, kernel_size=(1, 64), use_bias = False, padding='same', data_format="channels_first")(input_1)
    block2       = LayerNormalization()(block2)
    block2       = Activation(activation='elu')(block2)

    block2 = Concatenate(axis=1)([block1, block2])
    block2       = se_block(block2, 8)
    
    block3       = DepthwiseConv2D(kernel_size=(n_features, 1), depth_multiplier=2, use_bias = False, depthwise_constraint=max_norm(1.), data_format="channels_first")(block2)
    block3       = LayerNormalization()(block3)
    block3       = Activation(activation='elu')(block3)

    block3       = AveragePooling2D(pool_size=(1, 64), padding='same', data_format="channels_first")(block3)

    block5  = Flatten() (block3)
    block5       = Dense(n_outputs, kernel_constraint=max_norm(0.25))(block5)
    block5       = Activation(activation='softmax')(block5)

    return Model(inputs=input_1, outputs=block5)


def se_block(tensor, ratio=16):
    init = tensor
    channel_axis = 1 #if K.image_data_format() == "channels_first" else -1
    filters = tensor.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D(data_format="channels_first")(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    #if K.image_data_format() == 'channels_first':
    se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def plot_history(train_acc,val_acc, timestamp):
    plt.plot(train_acc, color ='blue', label='train')
    plt.plot(val_acc, color ='red', label = 'test')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc="upper left")
    plt.show()
    fig_file = f"curves/AccuracyCurve_{timestamp}.pdf"
    plt.savefig(fig_file)
