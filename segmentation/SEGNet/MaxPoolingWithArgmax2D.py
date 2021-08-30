import tensorflow as tf
from tensorflow.keras.layers import Layer


class MaxPoolingWithArgmax2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding="same", **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides

        ksize = [1, pool_size[0], pool_size[1], 1]
        padding = padding.upper()
        strides = [1, strides[0], strides[1], 1]
        output, argmax = tf.nn.max_pool_with_argmax( inputs, ksize=ksize, strides=strides, padding=padding)

        argmax = tf.cast(argmax, tf.float32)
        return [output, argmax]

    

def test():
    import tensorflow as tf
    x = tf.constant([   [1,2,3,4],
                        [1,3,4,5],
                        [2,3,6,8],
                        [7,3,6,1]])
    x = tf.reshape(x, [1, 4, 4, 1])
    print(x)
    print("=========================")
    p, m = MaxPoolingWithArgmax2D()(x)
    print(p)
    print(m)

if __name__ == "__main__":
    test()