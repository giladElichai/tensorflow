
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Dense, Flatten
from tensorflow.keras.models import Model

def conv_block(input_tensor, filters, n_blocks=2, kernel_size=3, padding='same'):

    x = input_tensor
    for _ in range(n_blocks):
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    return x



VGG_prm = {
  "VGG16": [(2, 64),(2, 128),(3, 256),(3, 512),(3, 512)],
  "VGG19": [(2, 64),(2, 128),(4, 256),(4, 512),(4, 512)],
}
# get vgg 16/19 model 
def GetVGGModel(input_shape, VGG_type="VGG16", out_channels=10):


    input_layer = Input(shape=input_shape, name='input_layer')

    conv_parm = VGG_prm[VGG_type]

    x = input_layer
    for n_blocks, n_filters in conv_parm:

        x = conv_block(x, filters=n_filters, n_blocks=n_blocks)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)

    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(out_channels, activation="softmax", name='out_layer')(x)

    name = '{}-model'.format(VGG_type)

    model = Model(inputs=input_layer, outputs=output_layer, name=name)
    return model


if __name__ == "__main__":

    model = GetVGGModel((128,128,3), 16, 10)

