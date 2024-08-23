from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Input, concatenate, Add, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf

def residual_block(input, filters, kernel_size=(3, 3, 3)):
    x = Conv3D(filters, kernel_size, padding='same')(input)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv3D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    shortcut = input
    if input.shape[-1] != filters:
        shortcut = Conv3D(filters, (1, 1, 1), padding='same')(input)

    x = Add()([shortcut, x])
    x = LeakyReLU()(x)
    return x

def my_res_unet_model(dim_in, Ncl, filters=[48, 64, 128], dropout_rate=0):

    input = Input(shape=(dim_in, dim_in, dim_in, 1))

    x = input
    down_layers = []

    # Encoder
    for filter in filters[:-1]:
        x = residual_block(x, filter)
        if dropout_rate > 0: x = Dropout(dropout_rate)(x)
        down_layers.append(x)
        x = MaxPooling3D((2, 2, 2))(x)

    # Bottleneck
    for _ in range(4):
        x = residual_block(x, filters[-1])

    # Decoder
    for filter, down_layer in zip(reversed(filters[:-1]), reversed(down_layers)):
        x = UpSampling3D(size=(2, 2, 2))(x)
        x = concatenate([x, down_layer])
        x = residual_block(x, filter)
        x = residual_block(x, filter)        
        if dropout_rate > 0: x = Dropout(dropout_rate)(x)

    output = Conv3D(Ncl, (1, 1, 1), padding='same', activation='softmax')(x)

    model = Model(input, output)
    return model
