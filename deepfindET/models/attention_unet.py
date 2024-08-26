from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, BatchNormalization, concatenate, Input, Activation, Multiply, Add, Lambda, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf

def conv_block(x, filters):
    x = Conv3D(filters, (3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(filters, (3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def attention_block(x, g, inter_channel):
    theta_x = Conv3D(inter_channel, (1, 1, 1), padding='same')(x)
    phi_g = Conv3D(inter_channel, (1, 1, 1), padding='same')(g)
    concat_xg = Add()([theta_x, phi_g])
    concat_xg = Activation('relu')(concat_xg)
    psi = Conv3D(1, (1, 1, 1), padding='same')(concat_xg)
    psi = Activation('sigmoid')(psi)
    psi_up = UpSampling3D(size=(2, 2, 2))(psi)
    psi_up = tf.repeat(psi_up, inter_channel, axis=-1)  # Use tf.repeat instead of K.repeat_elements
    out = Multiply()([x, psi_up])
    return out

def attention_unet(dim_in, Ncl, filters=[32, 48, 64], dropout_rate=0.2):
    input = Input(shape=(dim_in, dim_in, dim_in, 1))

    x = input
    down_layers = []

    # Encoder
    for filter in filters:
        x = conv_block(x, filter)
        down_layers.append(x)
        x = MaxPooling3D((2, 2, 2))(x)
        x = Dropout(dropout_rate)(x)

    # Bottleneck
    x = conv_block(conv_block(x, filters[-1]), filters[-1])

    # Decoder
    for filter, down_layer in zip(reversed(filters), reversed(down_layers)):
        x = UpSampling3D(size=(2, 2, 2))(x)
        x = attention_block(x, down_layer, filter)
        x = concatenate([x, down_layer])
        x = conv_block(x, filter)
        x = Dropout(dropout_rate)(x)

    output = Conv3D(Ncl, (1, 1, 1), padding='same', activation='softmax')(x)

    model = Model(inputs=input, outputs=output)
    return model