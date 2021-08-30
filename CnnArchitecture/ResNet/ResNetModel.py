
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Dense, Flatten, Add, AveragePooling2D
from tensorflow.keras.models import Model

def conv_block(input_tensor, filters, n_blocks=2, kernel_size=3, strides=1, padding='same', activation=True):
    x = input
    for _ in range(n_blocks):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(input_tensor)
        x = BatchNormalization()(x)
        if activation:
            x = Activation("relu")(x)
    return x


def residual_block(residual_input_data, filters, strides=1):
        
  # Hold input_x here for later processing
    identity_x = residual_input_data
        
    filter1,filter2,filter3 = filters
    s = strides if filter3 != identity_x.shape[-1] else 1

    conv_op_1 = conv_block(residual_input_data, filter1, n_blocks=1, kernel_size=1, strides=s, padding='valid')
    conv_op_2 = conv_block(conv_op_1, filter2, n_blocks=1, kernel_size=3, strides=1, padding='same')
    conv_op_3 = conv_block(conv_op_2, filter3, n_blocks=1, kernel_size=1, strides=1, padding='valid', activation=False)
  
    # Element-wise Addition
    if identity_x.shape[-1] != conv_op_3.shape[-1]:
        filter_n = conv_op_3.shape[-1]       
        identity_x = Conv2D(filters=filter_n, kernel_size=(1,1), strides=strides, padding='valid')(identity_x)
                
    output = Add()([identity_x, conv_op_3])
    output = Activation("relu")(output)

    return output


def ResNetModel(input_shape, type, layers_parm, out_channels=10):

    input_layer = Input(shape=input_shape, name='input_layer')

    x = input_layer
    x = conv_block(x, 64, n_blocks=1, kernel_size=7, strides=(2,2), padding='same')
    x = MaxPooling2D((3,3), strides=2, padding="same")(x)


    for n_blocks, filters, stride in layers_parm: 
        filters = (filters,filters,filters*4)
        for _ in range(n_blocks):
            x = residual_block(x, filters, stride)

    x = AveragePooling2D()(x)
    x = Flatten()(x)
    output_layer = Dense(out_channels, activation="softmax", name='out_layer')(x)

    name = 'ResNet{}-model'.format(type)

    model = Model(inputs=input_layer, outputs=output_layer, name=name)
    return model


def ResNet50Model(input_shape, out_channels=1000):

    layer_prm = [(3,64,1), (4,128,2), (6,256,2), (3,512,2)]
    model = ResNetModel(input_shape, 50, layer_prm, out_channels)

    return model

def ResNet101Model(input_shape, out_channels=1000):

    layer_prm = [(3,64,1), (4,128,2), (23,256,2), (3,512,2)]
    model = ResNetModel(input_shape, 101, layer_prm, out_channels)

    return model

def ResNet152Model(input_shape, out_channels=1000):

    layer_prm = [(3,64,1), (8,128,2), (36,256,2), (3,512,2)]
    model = ResNetModel(input_shape, 152, layer_prm, out_channels)

    return model


if __name__ == "__main__":

    model = ResNet50Model((224,224,3),  10)

    model.compile(optimizer='adam', loss='categorical_crossentropy' ,metrics=['accuracy'])

    import numpy as np
    m = np.random.rand(32,224,224,3)
    y = model.predict(m)
    print(y.shape)

