from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Flatten,
    Concatenate,
    Dense,
    Reshape,
    Conv2D,
    BatchNormalization,
    TimeDistributed,
    Conv2DTranspose,
    LSTM,
    Add
)

def my_conv(input_layer, filters, activation, time_distributed=False):
    if (time_distributed):
        l = TimeDistributed(Conv2D(filters, (3,3), padding='same', activation=activation))(input_layer)
        l = TimeDistributed(BatchNormalization())(l)
        return l
    else:
        l = Conv2D(filters, (3,3), padding='same', activation=activation)(input_layer)
        l = BatchNormalization()(l)
        return l

def my_downsampling(input_layer, filters):
    l = TimeDistributed(Conv2D(filters, (2,2), (2,2), activation='relu'))(input_layer)
    l = TimeDistributed(BatchNormalization())(l)
    return l

def my_conv_transpose(input_layer, skip_connection_layer):
    l = Conv2DTranspose(input_layer.shape[-1], (2,2), (2,2))(input_layer)
    skl = skip_connection_layer[:,-1,:,:,:] # extract features from last image
    l = Add()([l, skl])
    l = Activation('relu')(l)
    l = BatchNormalization()(l)
    return l


def my_model(len_c, len_p, len_t, nb_flow=2, map_height=32, map_width=32, external_dim=8, encoder_blocks=3, filters=[32,64,64,16]):

    main_inputs = []
    #ENCODER
    # input layer tx32x32x2
    input = Input(shape=((len_c+len_p+len_t, map_height, map_width, nb_flow)))
    main_inputs.append(input)
    x = input

    # build encoder blocks
    skip_connection_layers = []
    for i in range(0, encoder_blocks):        
        # conv + relu + bn
        x = my_conv(x, filters[i], 'relu', time_distributed=True)
        # append layer to skip connection list
        skip_connection_layers.append(x)
        # max pool
        x = my_downsampling(x, x.shape[-1])

    # last convolution tx4x4x16
    x = my_conv(x, filters[-1], 'relu')
    s = x.shape

    x = TimeDistributed(Flatten())(x)
    units = x.shape[-1]
    x = LSTM(units, return_sequences=True)(x)
    x = LSTM(units, return_sequences=True)(x)
    x = LSTM(units, return_sequences=True)(x)
    x = LSTM(units, return_sequences=False)(x)
    x = Reshape((s[2:]))(x)

    # build decoder blocks
    for i in reversed(range(0, encoder_blocks)):
        # conv + relu + bn
        x = my_conv(x, filters[i], 'relu')
        # conv_transpose + skip_conn + relu + bn
        x = my_conv_transpose(x, skip_connection_layers[i])

    # last convolution + tanh + bn 32x32x2
    output = Conv2D(nb_flow, (3,3), padding='same')(x)

    # merge external features
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(units=10, activation='relu')(external_input)
        h1 = Dense(units=nb_flow*map_height * map_width, activation='relu')(embedding)
        external_output = Reshape((map_height, map_width, nb_flow))(h1)
        output = Add()([external_output, output])
    
    output = Activation('tanh')(output)

    return Model(main_inputs, output)
