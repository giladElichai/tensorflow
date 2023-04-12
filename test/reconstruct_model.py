from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras import layers, Model

import numpy as np

def conv_block(input_tensor, filters, n_blocks=2, kernel_size=3, strides=1, padding='same', activation=True):
    x = input_tensor
    for _ in range(n_blocks):
        x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=True)(x)
        x = layers.BatchNormalization()(x)
        if activation:
            x = layers.Activation("relu")(x)
    return x


def main():

    resmodel = ResNet50(include_top=False, input_shape=(224,224,3))

    #print(resmodel.summary())

    tmp_model = Model(inputs=resmodel.layers[7].input, outputs=resmodel.output)

    input_layer = layers.Input(shape=(56, 224, 3))
    x = conv_block(input_layer, 64, n_blocks=1, kernel_size=7, strides=(1,2), padding='same')
    x = layers.MaxPooling2D((3,3), strides=(1,2), padding="same")(x)

    x = tmp_model(x) 
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(1, activation="softmax", name='out_layer')(x)
    model = Model(inputs=input_layer, outputs=output_layer, name="resnet50_reconstruct")

    print("pre")
    print(model.layers[1].get_weights())
    model.layers[1].set_weights(resmodel.layers[2].get_weights())
    print("post")
    print(model.layers[1].get_weights())

    print("pre")
    print(model.layers[2].get_weights())
    model.layers[2].set_weights(resmodel.layers[3].get_weights())
    print("post")
    print(model.layers[2].get_weights())

    x = np.random.rand(32,56,224,3)
    y = model.predict(x)
    print(y)
    x = 0

    pass

if __name__ == "__main__":
    main()