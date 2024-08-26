from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D
from tensorflow.keras.layers import Input, concatenate, Dropout
from tensorflow.keras.models import Model

def conv_block(input, filters, kernel_size=(3,3,3), activation='relu'):
    x = Conv3D(filters, kernel_size, padding='same', activation=activation)(input)
    x = Conv3D(filters, kernel_size, padding='same', activation=activation)(x)
    return x

def my_unet_model(dim_in, Ncl, filters=[32,48,64], dropout_rate=0):

    input = Input(shape=(dim_in, dim_in, dim_in, 1))
    
    x = input
    down_layers = []

    # Encoder
    for filter in filters[:-1]:
        x = conv_block(x, filter)
        x = Dropout(dropout_rate)(x)        
        down_layers.append(x)
        x = MaxPooling3D((2, 2, 2))(x)
    
    # Bottleneck
    x = conv_block( conv_block(x, filters[-1] ), filters[-1] )
    
    # Decoder
    for filter, down_layer in zip(reversed(filters[:-1]), reversed(down_layers)):
        x = UpSampling3D(size=(2, 2, 2))(x)
        x = Conv3D(filter, (2, 2, 2), padding='same', activation='relu')(x)
        x = concatenate([x, down_layer])
        x = conv_block(x, filter)

    output = Conv3D(Ncl, (1, 1, 1), padding='same', activation='softmax')(x)
    
    model = Model(input, output)
    return model