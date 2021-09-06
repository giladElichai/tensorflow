import tensorflow as tf
from tensorflow.keras.layers import Layer, concatenate, Activation

class YoloHeadLayer(Layer):
  def __init__(self, grid_size, num_boxes, num_classes, name):
    super(YoloHeadLayer, self).__init__(name=name)
    #self.target_shape = tuple(target_shape)
    self.S = grid_size
    self.B = num_boxes
    self.C = num_classes

  def get_config(self):
    config = super(YoloHeadLayer, self).get_config().copy()
    config.update({
     #   'target_shape': self.target_shape,
        'grid_size':self.S,
        'num_boxes': self.B, 
        'num_classes': self.C
    })
    return config


  def call(self, input):

    S =  self.S
    C = self.C
    B = self.B


    input  = tf.reshape(input, [-1, S, S, (B*5+C)] )

    # classes
    class_probs = Activation("softmax")(input[...,:C])

    # box1
    confs1 = Activation("sigmoid")( tf.reshape(input[..., C:C+1],[-1, S, S, 1]) )
    box1_xy = Activation("sigmoid")( tf.reshape(input[..., C+1:C+3],[-1, S, S, 2]) )
    box1_wh = Activation("sigmoid")( tf.reshape(input[..., C+3:C+5],[-1, S, S, 2]) ) * S
    
    #box2
    confs2 = Activation("sigmoid")( tf.reshape(input[..., C+5:C+6],[-1, S, S, 1]) )
    box2_xy = Activation("sigmoid")( tf.reshape(input[..., C+6:C+8],[-1, S, S, 2]) )
    box2_wh = Activation("sigmoid")( tf.reshape(input[..., C+8:],[-1, S, S, 2]) ) * S


    outputs = concatenate([class_probs, confs1, box1_xy, box1_wh, confs2, box2_xy, box2_wh])
    #outputs = concatenate([class_probs, confs1, box1, confs2, box2])
    

    return outputs


def test():

    tf.random.set_seed(5)
    x = tf.random.normal([4,1470], 0, 1, tf.float32)

    y_pred = YoloHeadLayer(7, 2, 20, name="output_layer")(x)
    x = 1 


if __name__ == "__main__":
    test()
