# ==================================================================================================
# ==================================================================================================
# Set up ATCNet Model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Dropout, BatchNormalization, Activation, Multiply
from tensorflow.keras.optimizers import Adam

def AttentionBlock(input_tensor):
    """Applies an attention mechanism over temporal features."""
    attention = Dense(input_tensor.shape[-1], activation='softmax')(input_tensor)
    return Multiply()([input_tensor, attention])

def ATCNet(input_shape, nb_classes, dropout_rate=0.5, num_filters=16, kernel_size=64):
    inputs = Input(shape=input_shape)

    # Temporal convolutional block
    x = Conv1D(filters=num_filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Attention mechanism
    x = AttentionBlock(x)

    # Flatten and fully connected layers
    x = Flatten()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(nb_classes, activation='softmax')(x)

    return Model(inputs, outputs)