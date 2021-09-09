from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Dense, Flatten, AveragePooling2D
from tensorflow.keras.layers import Add, Multiply, Lambda, UpSampling2D
from tensorflow.keras.models import Model


def conv_block(input_tensor, filters, n_blocks=2, kernel_size=3, strides=1, padding='same'):
    x = input_tensor
    for _ in range(n_blocks):        
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    return x

def pre_conv_block(input_tensor, filters, n_blocks=2, kernel_size=3, strides=1, padding='same'):
    #Pre-Activation conv block
    x = input_tensor
    for _ in range(n_blocks):        
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    return x


def residual_block( x, filters, strides=1 ):
    # Pre-Activation Identity ResUnit Bottleneck Architecture
    x_input = x
    
    filter1, filter2, filter3 = filters
    
    x = pre_conv_block(x, filter1, n_blocks=1, kernel_size=1, strides=strides, padding='valid')
    x = pre_conv_block(x, filter2, n_blocks=1, kernel_size=3, strides=1)
    x = pre_conv_block(x, filter3, n_blocks=1, kernel_size=1, strides=1)

    if x_input.shape[-1] != x.shape[-1]:
        x_input = Conv2D(x.shape[-1], 1, strides=strides, padding='same')(x_input)
        
    output = Add()([x_input, x])

    return output


def trunk_branch( trunk_input, filters, t=2):
    
    x = trunk_input
    for _ in range(t):
        x = residual_block(x, filters=filters)

    return x


def mask_branch( mask_input, filters, m=3, r=1 ):
    # r = num of residual units between adjacent pooling layers, default=1
    # m = num max pooling / linear interpolations to do

    x = mask_input
    
    skip = {}
    for i in reversed(range(m)):

        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

        for _ in range(r):
            x = residual_block(x, filters=filters)
        
        skip[i] = x


    for i in range(m):

        if i > 0:
            sk_x = residual_block(skip[i], filters=filters)
            x = Add()([x, sk_x])

        for _ in range(r):
            x = residual_block(x, filters=filters)

        x = UpSampling2D(size=(2, 2))(x)

        
    x = pre_conv_block(x, filters[2], n_blocks=1, kernel_size=1, strides=1)
    x = pre_conv_block(x, 1, n_blocks=1, kernel_size=1, strides=1)

    name= f"mask_layer_op_{4-m}"
    out = Activation('sigmoid', name=name)(x)

    return out

def attention_residual_learning( mask_input, trunk_input ):

    Mx = Lambda(lambda x: 1 + x)(mask_input) # 1 + mask
    return Multiply()([Mx, trunk_input]) # M(x) * T(x)


def attention_module( attention_input, filters, m=3, p=1, t=2, r=1):

    x = attention_input
    for _ in range(p):
        x = residual_block(x, filters=filters)

    trunk_out = trunk_branch(trunk_input=x, filters=filters, t=t)
    mask_out = mask_branch(mask_input=x, filters=filters, m=m, r=r)

    x = attention_residual_learning(mask_input=mask_out, trunk_input=trunk_out)

    for _ in range(p):
        x = residual_block(x, filters=filters)

    return x


def ResidualAttentionModel( input_shape, num_classes, p=1, t=2, r=1):

    input_layer = Input(shape=input_shape)
    
    x = input_layer
    x = Conv2D(64, 7, strides=(2,2), padding='same' )(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    
    filters=[64, 64, 256]
    x = residual_block(x, filters)
    att_model_out1 = attention_module(x, filters=filters, m=3, p=p, t=t, r=r)
    
    filters = [128, 128, 512]
    x = residual_block(att_model_out1, filters, strides=2)
    att_model_out2 = attention_module(x, filters=filters, m=2, p=p, t=t, r=r)

    filters = [256, 256, 1024]
    x = residual_block(att_model_out2, filters, strides=2)
    att_model_out3 = attention_module(x, filters=filters, m=1, p=p, t=t, r=r)

    filters = [512, 512, 2048]
    x = residual_block(att_model_out3, filters, strides=2)
    x = residual_block(x, filters)
    x = residual_block(x, filters)
    
    
    x = AveragePooling2D(x.shape[1])(x)
    x = Flatten()(x)

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    output_layer = Dense(num_classes, activation="softmax", name='out_layer')(x)

    name = 'ResidualAttentionModel'
    model = Model(inputs=input_layer, outputs=output_layer, name=name)

    model.compile(optimizer='adam', loss='categorical_crossentropy' ,metrics=['accuracy'])

    return model


if __name__ == "__main__":

    model = ResidualAttentionModel((224,224,3), 1000)

    import numpy as np
    m = np.random.rand(32,224,224,3)
    y = model.predict(m)
    print(y.shape)
