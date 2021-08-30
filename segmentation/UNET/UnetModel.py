
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, BatchNormalization, Activation, Dropout, UpSampling2D
from tensorflow.keras.models import Model

def conv_block(input_tensor, filters, n_blocks=2, kernel_size=3, padding='same', initializer="he_normal"):

    x = input
    for _ in range(n_blocks):
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, kernel_initializer=initializer)(input_tensor)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    return x


def deconv_block(input_tensor, skip_tensor, nfilters, kernel_size=3, padding='same', strides=(2, 2)):

    input_tensor = UpSampling2D()(input_tensor)
    y = concatenate([input_tensor, skip_tensor], axis=3)
    y = conv_block(y, nfilters, n_blocks=2, kernel_size=kernel_size)
    return y


def GetUnetModel(input_shape, kernel_size=3, init_filters=64, n_levels=4, out_channels=1):
    skips = {}

    input_layer = Input(shape=input_shape, name='image_input')
    # down
    x = input_layer
    for level in range(n_levels):
        n_filters = init_filters * 2**level
        x = conv_block(x, filters=n_filters, n_blocks=2, kernel_size=kernel_size)
        skips[level] = x
        x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)

    #bottom
    n_filters = init_filters * 2**n_levels
    x = conv_block(x, filters=n_filters, n_blocks=1, kernel_size=kernel_size)
    x = Dropout(0.5)(x)

    #up
    for level in reversed(range(n_levels)):
        n_filters = init_filters * 2**level
        x = conv_block(x, filters=n_filters, n_blocks=1, kernel_size=kernel_size)
        x = deconv_block(x, skips[level], n_filters)
        x = Dropout(0.1)(x)

    x = Dropout(0.5)(x)
    # output head
    activation = 'sigmoid' if out_channels == 1 else 'softmax'
    output_layer = Conv2D(out_channels, kernel_size=1, activation=activation, padding='same', name='mask')(x)

    name = 'UNet-L{}-F{}'.format(n_levels, init_filters)
    model = Model(inputs=input_layer, outputs=output_layer, name=name)
    return model


if __name__ == "__main__":

    model = GetUnetModel((256,256,3), 3, 64, 4, 32)

