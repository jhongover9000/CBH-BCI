from keras.models import Model
from keras.layers import (Input, Conv2D, DepthwiseConv2D, AveragePooling2D, SeparableConv2D, 
                          BatchNormalization, Activation, Dropout, Flatten, Dense, GlobalAveragePooling1D, 
                          Concatenate, Lambda, Permute, Reshape)
from keras.layers import GlobalAveragePooling2D, Softmax
from keras import backend as K
from models.attention_models import se_block, cbam_block, mha_block
from models.models import TCN_block_


def ATCNet_(nb_classes, Chans=22, Samples=1000, dropoutRate=0.5,
            kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25,
            dropoutType='Dropout', attention_type=None,
            n_windows=5, fusion_type='average', from_logits=False):

    input_main = Input(shape=(1, Chans, Samples))
    block1 = Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(input_main)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D,
                              depthwise_constraint=None, padding='valid')(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(dropoutRate)(block2)

    # Reshape to (batch_size, time_steps, features)
    reshaped = Reshape((block2.shape[-1], -1))(block2)
    reshaped = Permute((2, 1))(reshaped)  # Now (batch, time, features)

    # Create temporal windows
    step = reshaped.shape[1] // n_windows
    window_outputs = []
    for i in range(n_windows):
        st = i * step
        end = (i + 1) * step if i < n_windows - 1 else reshaped.shape[1]
        window = Lambda(lambda x, s=st, e=end: x[:, s:e, :])(reshaped)

        # Attention block (optional)
        if attention_type == 'se':
            window = se_block(window)
        elif attention_type == 'cbam':
            window = cbam_block(window)
        elif attention_type == 'mha':
            window = mha_block(window)

        # TCN block
        window = TCN_block_(window)
        window_outputs.append(window)

    # Fusion of window outputs
    if fusion_type == 'average':
        sw_concat = Lambda(lambda x: K.mean(K.stack(x, axis=1), axis=1))(window_outputs)
    elif fusion_type == 'concat':
        sw_concat = Concatenate(axis=-1)(window_outputs)
    else:
        raise ValueError("fusion_type must be 'average' or 'concat'")

    # Final dense output
    out = Dense(nb_classes)(sw_concat)
    out = Activation('linear' if from_logits else 'softmax', name='output')(out)

    return Model(inputs=input_main, outputs=out)
