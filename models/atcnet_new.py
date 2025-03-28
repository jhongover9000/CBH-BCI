import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, DepthwiseConv2D,
    Activation, AveragePooling2D, Flatten, Dense, Dropout,
    Permute, Lambda, Concatenate
)
from tensorflow.keras.regularizers import L2
from models.models import Conv_block_, attention_block, TCN_block_

def ATCNet_(n_classes, in_chans=22, in_samples=1125, n_windows=5, attention='mha', 
           eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3, 
           tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3, 
           tcn_activation='elu', fuse='average'):
    
    """ ATCNet with Batch Normalization (BN) integrated """

    input_1 = Input(shape=(1, in_chans, in_samples))
    input_2 = Permute((3, 2, 1))(input_1)

    dense_weightDecay = 0.5  
    conv_weightDecay = 0.009
    conv_maxNorm = 0.6
    from_logits = False

    numFilters = eegn_F1
    F2 = numFilters * eegn_D

    # EEGNet-style convolution block
    block1 = Conv_block_(input_layer=input_2, F1=eegn_F1, D=eegn_D,
                         kernLength=eegn_kernelSize, poolSize=eegn_poolSize,
                         weightDecay=conv_weightDecay, maxNorm=conv_maxNorm,
                         in_chans=in_chans, dropout=eegn_dropout)
    
    block1 = BatchNormalization()(block1)  # BN after Conv Block
    block1 = Activation('relu')(block1)

    block1 = Lambda(lambda x: x[:, :, -1, :], output_shape=(block1.shape[1], block1.shape[-1]))(block1)  # Ensure valid shape

    # **Fix: Ensure time dimension is large enough**
    time_dim = block1.shape[1]
    if time_dim < n_windows:
        print(f"Warning: Time dimension ({time_dim}) is smaller than n_windows ({n_windows}). Adjusting...")
        n_windows = max(1, time_dim)  # Set n_windows to a valid value

    # Sliding window mechanism
    sw_concat = []
    for i in range(n_windows):
        st = i
        end = min(time_dim, i + (time_dim // n_windows))  # Ensure valid index

        if end <= st:  # Prevent invalid range
            print(f"Skipping window {i} due to invalid range: start={st}, end={end}")
            continue

        block2 = block1[:, st:end, :]

        # Attention mechanism
        if attention is not None:
            if attention in ['se', 'cbam']:
                block2 = Permute((2, 1))(block2)
                block2 = attention_block(block2, attention)
                block2 = Permute((2, 1))(block2)
            else:
                block2 = attention_block(block2, attention)

        block2 = BatchNormalization()(block2)  # BN before TCN

        # Temporal Convolutional Network (TCN)
        block3 = TCN_block_(input_layer=block2, input_dimension=F2, depth=tcn_depth,
                            kernel_size=tcn_kernelSize, filters=tcn_filters, 
                            weightDecay=conv_weightDecay, maxNorm=conv_maxNorm,
                            dropout=tcn_dropout, activation=tcn_activation)
        
        block3 = BatchNormalization()(block3)  # BN in TCN
        block3 = Lambda(lambda x: x[:, -1, :], output_shape=(block3.shape[-1],))(block3)  # Fix shape extraction

        # Fuse sliding window outputs
        if fuse == 'average':
            sw_concat.append(Dense(n_classes, kernel_regularizer=L2(dense_weightDecay))(block3))
        elif fuse == 'concat':
            sw_concat = Concatenate()([sw_concat, block3]) if i != 0 else block3

    # **Fix: Ensure sw_concat is not empty**
    if len(sw_concat) == 0:
        raise ValueError(f"Sliding window failed: No valid windows in the input sequence. Time dim: {time_dim}, n_windows: {n_windows}")

    # Output layer
    if fuse == 'average':
        if len(sw_concat) > 1:
            sw_concat = tf.keras.layers.Average()(sw_concat[:])
        else:
            sw_concat = sw_concat[0]
    elif fuse == 'concat':
        sw_concat = Dense(n_classes, kernel_regularizer=L2(dense_weightDecay))(sw_concat)

    out = Activation('linear' if from_logits else 'sigmoid', name='output')(sw_concat)

    return Model(inputs=input_1, outputs=out)
