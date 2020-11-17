import tensorflow as tf
from keras.layers import Input, Conv2D
from keras.models import Model
from keras.optimizers import Adam

from layers.ResidualMultiplicativeBlock import ResidualMultiplicativeBlock as rmb
from layers.CascadeMultiplicativeUnit import CascadeMultiplicativeUnit as cmu


def predcnn(params, mask_true, num_hidden, filter_size, seq_length=20, input_length=10):
    encoder_length = params['encoder_length']
    decoder_length = params['decoder_length']
    map_height = params['map_height']
    map_width = params['map_width']
    channels = params['channels']

    with tf.variable_scope('predcnn'):
        # encoder
        main_input = Input(shape=(input_length, map_height, map_width, channels))
        encoder_output = []
        for i in range(input_length):
            reuse = bool(encoder_output)
            ims = main_input[:,i]
            input = resolution_preserving_cnn_encoders(ims, num_hidden, filter_size, encoder_length, reuse)
            encoder_output.append(input)

        # predcnn & decoder
        output = []
        for i in range(seq_length - input_length):
            reuse = bool(output)
            out = predcnn_perframe(encoder_output[i:i+input_length], num_hidden, filter_size, input_length, reuse)
            out = cnn_docoders(out, num_hidden, filter_size, channels, decoder_length, reuse)
            output.append(out)

            # ims = mask_true[:, 0] * images[:, input_length + i] + (1 - mask_true[:, 0]) * out
            # input = resolution_preserving_cnn_encoders(ims, num_hidden, filter_size, encoder_length, reuse=True)
            # encoder_output.append(input)

    # # transpose output and compute loss
    # gen_images = tf.stack(output)
    # # [batch_size, seq_length, height, width, channels]
    # gen_images = tf.transpose(gen_images, [1, 0, 2, 3, 4])
    # loss = tf.nn.l2_loss(gen_images - images[:, input_length:])

    # return [gen_images, loss]
    return Model(main_input, output)


def resolution_preserving_cnn_encoders(x, num_hidden, filter_size, encoder_length, reuse):
    with tf.variable_scope('resolution_preserving_cnn_encoders', reuse=reuse):
        x = Conv2D(num_hidden, filter_size, padding='same', activation=None,
                  #  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                   name='input_conv')(x)
        for i in range(encoder_length):
            x = rmb('residual_multiplicative_block_'+str(i+1), num_hidden // 2, filter_size)(x)
        return x


def predcnn_perframe(xs, num_hidden, filter_size, input_length, reuse):
    with tf.variable_scope('frame_prediction', reuse=reuse):
        for i in range(input_length-1):
            temp = []
            for j in range(input_length-i-1):
                h1 = xs[j]
                h2 = xs[j+1]
                h = cmu('causal_multiplicative_unit_'+str(i+1), num_hidden, filter_size)(h1, h2, stride=False, reuse=bool(temp))
                temp.append(h)
            xs = temp
        return xs[0]


def cnn_docoders(x, num_hidden, filter_size, output_channels, decoder_length, reuse):
    with tf.variable_scope('cnn_decoders', reuse=reuse):
        for i in range(decoder_length):
            x = rmb('residual_multiplicative_block_'+str(i+1), num_hidden // 2, filter_size)(x)
        x = Conv2D(output_channels, filter_size, padding='same',
                  #  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                   name='output_conv')(x)
        return x
