
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, BatchNormalization, Activation, Dropout, UpSampling2D, Add
from tensorflow.keras.models import Model

def conv_block(input_tensor, filters, kernel, strides, padding='same', activation="relu" ):

    x = Conv2D(filters=filters, kernel_size=kernel, strides=strides, padding=padding, use_bias=False)(input_tensor)
    x = BatchNormalization()(x)
    if activation:
        x = Activation(activation)(x)

    return x

def conv_transpose_block(input_tensor, filters, kernel, strides, padding='same', activation="relu" ):

    x = Conv2DTranspose(filters=filters, kernel_size=kernel, strides=strides, padding=padding, use_bias=False)(input_tensor)
    x = BatchNormalization()(x)
    if activation:
        x = Activation(activation)(x)

    return x



def residual_block( residual_input, filters, downsample=False):

    residual = residual_input

    strides = (2,2) if downsample else (1,1) 

    x = conv_block( residual_input, filters, kernel=3, strides=strides)
    x = conv_block( x, filters, kernel=3, strides=(1,1), activation="")

    if downsample:
        residual = conv_block( residual, filters, kernel=1, strides=strides, activation="")


    x = Add()([residual, x])
    x = Activation("relu")(x)

    return x


def encoder_block( input_tensor, filters):

    x = residual_block(input_tensor, filters, downsample=True)
    x = residual_block(x, filters, downsample=False)

    return x

def decoder_block( input_tensor, filters):

    m_filters =  input_tensor.shape[3] / 4

    x = conv_block( input_tensor, m_filters, kernel=1, strides=(1,1))
    x = conv_transpose_block(x, m_filters, kernel=3, strides=(2,2))
    x = conv_block( x, filters, kernel=1, strides=(1,1))

    return x


def GetLinkNetModel(input_shape, init_filters=64, n_levels=4, out_channels=1):
    
    skips = {}

    input_layer = Input(shape=input_shape, name='input_layer')
    x = input_layer

    # init conv
    x = conv_block(x, filters=64, kernel=7, strides =(2,2))
    x = MaxPooling2D((3,3), strides=2, padding="same")(x)

    # down
    for level in range(n_levels):
        n_filters = init_filters * 2**level
        x = encoder_block(x, n_filters)
        skips[level] = x

    x = Dropout(0.5)(x)

    # up
    for level in reversed(range(n_levels-1)):
        n_filters = init_filters * 2**(level)
        x = decoder_block(x, n_filters)
        x = Add()([skips[level], x])
        x = Dropout(0.1)(x)
    x = decoder_block(x, init_filters)

    x = Dropout(0.5)(x)

    # classifier 
    x = conv_transpose_block(x, 32, kernel=3, strides=(2,2))
    x = conv_block( x, 32, kernel=3, strides=(1,1))
    x = Dropout(0.3)(x)
    x = Conv2DTranspose(out_channels, kernel_size=3, strides=(2,2), padding='same')(x)

    # output head
    activation = 'sigmoid' if out_channels == 1 else 'softmax'
    output_layer = Activation(activation, name='output_mask')(x)

    name = 'Linknet-L{}-F{}'.format(n_levels, init_filters)
    model = Model(inputs=input_layer, outputs=output_layer, name=name)
    return model


if __name__ == "__main__":

    model = GetLinkNetModel((256,256,3), 64, 4, 32)

