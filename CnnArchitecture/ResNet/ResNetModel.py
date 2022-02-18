
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Dense, Add, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def conv_block(input_tensor, filters, n_blocks=2, kernel_size=3, strides=1, padding='same', activation=True):
    x = input_tensor
    for _ in range(n_blocks):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(x)
        x = BatchNormalization()(x)
        if activation:
            x = Activation("relu")(x)
    return x


def residual_block(residual_input_data, filters, downsample=False):
        
  # Hold input_x here for later processing
    identity_x = residual_input_data
        
    filter1,filter2,filter3 = filters
    strides = 2 if downsample else 1

    conv_op_1 = conv_block(residual_input_data, filter1, n_blocks=1, kernel_size=1, strides=strides, padding='valid')
    conv_op_2 = conv_block(conv_op_1, filter2, n_blocks=1, kernel_size=3, strides=1, padding='same')
    conv_op_3 = conv_block(conv_op_2, filter3, n_blocks=1, kernel_size=1, strides=1, padding='valid', activation=False)
  
    if downsample or identity_x.shape[-1] != filter3:      
        identity_x = Conv2D(filters=filter3, kernel_size=(1,1), strides=strides, padding='valid')(identity_x)

    # Element-wise Addition       
    output = Add()([identity_x, conv_op_3])
    output = Activation("relu")(output)

    return output


def ResNetModel(input_shape, type, layers_parm, out_channels=10):

    input_layer = Input(shape=input_shape, name='input_layer')

    x = input_layer
    x = conv_block(x, 64, n_blocks=1, kernel_size=7, strides=(2,2), padding='same')
    x = MaxPooling2D((3,3), strides=2, padding="same")(x)


    for n_blocks, filters, downsample in layers_parm: 
        filters = (filters,filters,filters*4)
        for i in range(n_blocks):
            to_downsample = downsample if i == 0 else False
            x = residual_block(x, filters, to_downsample)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    
    output_layer = Dense(out_channels, activation="softmax", name='out_layer')(x)

    name = 'ResNet{}-model'.format(type)

    model = Model(inputs=input_layer, outputs=output_layer, name=name)
    return model


def ResNet50Model(input_shape, out_channels=1000):

    layer_prm = [(3,64,False), (4,128,True), (6,256,True), (3,512,True)]
    model = ResNetModel(input_shape, 50, layer_prm, out_channels)

    return model

def ResNet101Model(input_shape, out_channels=1000):

    layer_prm = [(3,64,False), (4,128,True), (23,256,True), (3,512,True)]
    model = ResNetModel(input_shape, 101, layer_prm, out_channels)

    return model

def ResNet152Model(input_shape, out_channels=1000):

    layer_prm = [(3,64,False), (8,128,True), (36,256,True), (3,512,True)]
    model = ResNetModel(input_shape, 152, layer_prm, out_channels)

    return model


if __name__ == "__main__":

    model = ResNet50Model((224,224,3),  10)

    model.compile(optimizer='adam', loss='categorical_crossentropy' ,metrics=['accuracy'])

    import numpy as np
    m = np.random.rand(32,224,224,3)
    y = model.predict(m)
    print(y.shape)

