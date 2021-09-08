
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Activation, Dropout, Dense, Flatten, concatenate, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def conv_block(input_tensor, filters, n_blocks=1, kernel_size=3, strides=1, padding='same'):

    x = input_tensor
    for _ in range(n_blocks):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    return x


def inception(x, f_1x1, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5, f_pool, name = None):

    out_1x1 = Conv2D(f_1x1, (1, 1), padding='same', activation='relu')(x)

    out_3x3 = Conv2D(f_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    out_3x3 = Conv2D(f_3x3, (1, 1), padding='same', activation='relu')(out_3x3)

    out_5x5 = Conv2D(f_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    out_5x5 = Conv2D(f_5x5, (1, 1), padding='same', activation='relu')(out_5x5)

    out_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    out_pool = Conv2D(f_pool, (1, 1), padding='same', activation='relu')(out_pool)

    out = concatenate([out_1x1, out_3x3, out_5x5, out_pool], axis=3, name=name)

    return out

def aux_out_branch(x, out_channels, name=None):

    aux_out = AveragePooling2D((5, 5), strides=3)(x)
    aux_out = Conv2D(128, 1, padding='same', activation='relu')(aux_out)
    aux_out = Flatten()(aux_out)
    aux_out = Dense(1024, activation='relu')(aux_out)
    aux_out = Dropout(0.7)(aux_out) 
    aux_out = Dense(out_channels, activation='softmax', name=name)(aux_out)

    return aux_out

def GetInceptionV1Model(input_shape, out_channels=1000):

    input_layer = Input(shape=input_shape, name='input_layer')

    #x = Conv2D(64, 7, strides=2, padding='same', activation='relu')(input_layer)
    x = conv_block(input_layer, 64, n_blocks=1, kernel_size=7, strides=2, padding="same")
    x = MaxPooling2D(3, strides=2)(x)
    #x = Conv2D(64, 1, strides=1, padding='same', activation='relu')(x)
    x = conv_block(x, 64, n_blocks=1, kernel_size=1, strides=1, padding="same")
    #x = Conv2D(192, 3, strides=1, padding='same', activation='relu')(x)
    x = conv_block(x, 192, n_blocks=1, kernel_size=3, strides=1, padding="same")
    x = MaxPooling2D(3, strides=2)(x)

    out_3a = inception(x, f_1x1=64, f_3x3_reduce=96, f_3x3=128, f_5x5_reduce=16, f_5x5=32, f_pool=32, name="Inception_3a" )
    out_3b = inception(out_3a, f_1x1=128, f_3x3_reduce=128, f_3x3=192, f_5x5_reduce=32, f_5x5=96, f_pool=64, name="Inception_3b" )
    x = MaxPooling2D(3, strides=2)(out_3b)

    out_4a = inception(x, f_1x1=192, f_3x3_reduce=96, f_3x3=208, f_5x5_reduce=16, f_5x5=48, f_pool=64, name="Inception_4a" )
    out_4b = inception(out_4a, f_1x1=160, f_3x3_reduce=112, f_3x3=224, f_5x5_reduce=24, f_5x5=64, f_pool=64, name="Inception_4b" )
    out_4c = inception(out_4b, f_1x1=128, f_3x3_reduce=128, f_3x3=256, f_5x5_reduce=24, f_5x5=64, f_pool=64, name="Inception_4c" )
    out_4d = inception(out_4c, f_1x1=112, f_3x3_reduce=144, f_3x3=288, f_5x5_reduce=32, f_5x5=64, f_pool=64, name="Inception_4d" )
    out_4e = inception(out_4d, f_1x1=256, f_3x3_reduce=160, f_3x3=320, f_5x5_reduce=32, f_5x5=128, f_pool=128, name="Inception_4e" )
    x = MaxPooling2D(3, strides=2)(out_4e)

    out_5a = inception(x, f_1x1=256, f_3x3_reduce=160, f_3x3=320, f_5x5_reduce=32, f_5x5=128, f_pool=128, name="Inception_5a" )
    out_5b = inception(out_5a, f_1x1=384, f_3x3_reduce=192, f_3x3=384, f_5x5_reduce=48, f_5x5=128, f_pool=128, name="Inception_5b" )
    x = GlobalAveragePooling2D()(out_5b)
    x = Dropout(0.4)(x)
    main_out = Dense(out_channels, activation='softmax')(x)

    aux1_out = aux_out_branch(out_4a, out_channels, "aux1_out")
    aux2_out = aux_out_branch(out_4d, out_channels, "aux2_out")

    name = 'InceptionV1-O{}-model'.format(out_channels)
    outputs=[main_out, aux1_out, aux2_out]
    model = Model(inputs=input_layer, outputs=outputs, name=name)

    loss=["sparse_categorical_crossentropy", "sparse_categorical_crossentropy", "sparse_categorical_crossentropy"]
    model.compile(optimizer='adam', loss=loss, loss_weights=[1, 0.3, 0.3], metrics=['accuracy'])

    return model


if __name__ == "__main__":

    model = GetInceptionV1Model((150,150,1), 100)

    import numpy as np 
    data = np.random.rand(3,224,224,1)

    pred = model.predict(data)

    x = 1 
