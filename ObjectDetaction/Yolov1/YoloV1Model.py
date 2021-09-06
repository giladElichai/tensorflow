import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Reshape, Activation, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import l2



from YoloHeadLayer import YoloHeadLayer

image_Size = (448,448)

def GetYoloV1Model(grid_size, num_boxes, num_classes, input_shape = (448, 448, 3)):

    out_size = grid_size*grid_size*(num_boxes*5+ num_classes)

    lrelu = LeakyReLU(alpha=0.1)

    input_layer = Input(shape=input_shape, name='input_layer')
    x = Conv2D( 64, (7,7), strides=(2,2), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4) )(input_layer)
    x = MaxPooling2D( (2,2), strides=(2,2), padding='same' )(x)

    x = Conv2D( 192, (3,3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4) )(x)
    x = MaxPooling2D( (2,2), strides=(2,2), padding='same' )(x)

    x = Conv2D( 128, (1,1), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4) )(x)
    x = Conv2D( 256, (3,3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4) )(x)
    x = Conv2D( 256, (1,1), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4) )(x)
    x = Conv2D( 512, (3,3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4) )(x)
    x = MaxPooling2D( (2,2), strides=(2,2), padding='same' )(x)

    for i in range(4):
        x = Conv2D( 256, (1,1), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4) )(x)
        x = Conv2D( 512, (3,3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4) )(x)
    
    x = Conv2D( 512, (1,1), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4) )(x)
    x = Conv2D( 1024, (3,3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4) )(x)
    x = MaxPooling2D( (2,2), strides=(2,2), padding='same' )(x)

    for i in range(2):
        x = Conv2D( 512, (1,1), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4) )(x)
        x = Conv2D( 1024, (3,3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4) )(x)
    
    x = Conv2D( 1024, (3,3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4) )(x)
    x = Conv2D( 1024, (3,3), strides=(2,2), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4) )(x)
    

    x = Conv2D( 1024, (3,3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4) )(x)
    x = Conv2D( 1024, (3,3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4) )(x)

    x = Flatten()(x)
    x = Dense(4096)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5)(x)
    x = Dense(out_size, activation='relu')(x)

    output = YoloHeadLayer(grid_size, num_boxes, num_classes, name="output_layer")(x)

    name = 'InceptionV1-model'
    model = Model(inputs=input_layer, outputs=output, name=name)

    return model



def test():

    # model = GetYoloV1Model(7, 2, 20, input_shape = (448, 448, 3))

    # import numpy as np
    # x = np.random.rand(4,448,448,3)
    # model(x)

    # model.summary()

    x = 1 


if __name__ == "__main__":
    test()
