
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Model

import tensorflow_addons as tfa
from MaxPoolingWithArgmax2D import MaxPoolingWithArgmax2D


def conv_block(input_tensor, filters, n_blocks=2, kernel_size=3, padding='same', initializer="glorot_uniform"):

    x = input
    for _ in range(n_blocks):
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, kernel_initializer=initializer)(input_tensor)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    return x


def GetSegnetModel(input_shape, convBlocks, kernel_size=3, out_channels=1):
    pools_indices = {}

    l = len(convBlocks)
    input_layer = Input(shape=input_shape, name='image_input')
    # down

    x = input_layer
    for i, conv in enumerate(convBlocks):
        n_blocks, filters = conv
        x = conv_block(x, filters=filters, n_blocks=n_blocks, kernel_size=kernel_size)
        pool, indices = MaxPoolingWithArgmax2D(pool_size=(2, 2))(x)
        pools_indices[i] = indices
        x = pool 
        x = Dropout(0.1)(x)

    x = Dropout(0.5)(x)

    #up
    for i, conv in enumerate(reversed(convBlocks)):
        n_blocks, filters = conv
        if i > 0:
            x = conv_block(x, filters=filters, n_blocks=1, kernel_size=kernel_size)
        mask = pools_indices[l-i-1]
        x = tfa.layers.MaxUnpooling2D()(x, mask)
        x = conv_block(x, filters=filters, n_blocks=n_blocks, kernel_size=kernel_size)
        
    x = Dropout(0.5)(x)
    # output head
    activation = 'sigmoid' if out_channels == 1 else 'softmax'
    output_layer = Conv2D(out_channels, kernel_size=1, activation=activation, padding='same', name='mask')(x)

    name = 'Seget-{}'.format(123)
    model = Model(inputs=input_layer, outputs=output_layer, name=name)
    return model


if __name__ == "__main__":


    convs = [(2,64), (2,128), (3,256), (3,512), (3,512)]
    model = GetSegnetModel((256,256,3), convs, 3, 32)


    
    
