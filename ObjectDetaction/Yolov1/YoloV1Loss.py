import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import  concatenate


def box_corner_points( input ):

    xy = input[...,:2]
    wh = input[...,2:]

    p0 = xy - (wh / 2)
    p1 = xy + (wh / 2)

    return p0[...,:1], p0[...,1:2], p1[...,:1], p1[...,1:2]


def intersection_over_union(boxes_labels, boxes_preds):


    box1_x1, box1_y1, box1_x2, box1_y2 = box_corner_points(boxes_preds)
    box2_x1, box2_y1, box2_x2, box2_y2 = box_corner_points(boxes_labels)


    x1 = tf.maximum(box1_x1, box2_x1)
    y1 = tf.maximum(box1_y1, box2_y1)
    x2 = tf.minimum(box1_x2, box2_x2)
    y2 = tf.minimum(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = tf.clip_by_value(x2 - x1, 0, 448) * tf.clip_by_value(y2 - y1, 0, 448)

    box1_area = tf.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = tf.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    iou = intersection / (box1_area + box2_area - intersection + 1e-6)

    return iou




def yolo_loss(y_true, y_pred):

    mse = tf.keras.losses.MeanSquaredError()
    
    label_class = y_true[..., :20]  # bs * 7 * 7 * 20
    label_box = y_true[..., 21:25]  # bs * 7 * 7 * 4
    response_mask = y_true[..., 20:21]  # bs * 7 * 7 * 1
    

    predict_class = y_pred[..., :20]  # bs * 7 * 7 * 20
    predict_box1 =  y_pred[..., 21:25]  # bs * 7 * 7 * 4
    predict_box2 = y_pred[..., 26:] # bs * 7 * 7 * 4

    iou_b1 = intersection_over_union(label_box, predict_box1)
    iou_b2 = intersection_over_union(label_box, predict_box2)
    ious = concatenate([tf.expand_dims(iou_b1, 0), tf.expand_dims(iou_b2, 0)], axis=0)


    iou_maxes, bestbox = K.max(ious, axis=0), tf.cast(K.argmax(ious, axis=0), tf.float32)

    #print("bestbox: ", bestbox.numpy())

    box_pred = response_mask * ( (1 - bestbox) * predict_box1 + bestbox * predict_box2 )  
    box_label = response_mask * label_box


    box_loss = mse( box_pred, box_label )
    #print("box_loss: ", box_loss.numpy())
    
    #===============================================================================
    # object loss
    pred_box = (
            bestbox * y_pred[..., 25:26] + (1 - bestbox) * y_pred[..., 20:21]
        )

    object_loss = mse( (response_mask * pred_box), (response_mask * response_mask ) )

    #print("object_loss: ", object_loss.numpy())

    #===============================================================================
    # no object loss
    no_object_loss = mse(
        (1 - response_mask) * y_pred[..., 20:21],
        (1 - response_mask) * response_mask
    )

    no_object_loss += mse(
        (1 - response_mask) * y_pred[..., 25:26],
        (1 - response_mask) * response_mask
    )

    #print("no_object_loss: ", no_object_loss.numpy())

    #====================================================
    #class loss
    class_loss = mse(
        response_mask * predict_class,
        response_mask * label_class,
    )

    #print("class_loss: ", class_loss.numpy())

    #====================================================
    #total loss

    lambda_noobj = 0.5
    lambda_coord = 5

    loss = (
            lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

    #print("loss: ", loss.numpy())

    return loss



def test():
    x =1 

    # from DataLoader import DataGenerator
    # from YoloV1Model import GetYoloV1Model

    # batch_size = 1
    # csv_file = "D:\\programing\\DataSets\\ObjectDetection\\pascalVoc\\train.csv"
    # img_dir = "D:\\programing\\DataSets\\ObjectDetection\\pascalVoc\\images"
    # labels_dir = "D:\\programing\\DataSets\\ObjectDetection\\pascalVoc\\labels"

    # train_dg = DataGenerator(csv_file, img_dir, labels_dir, (448, 448), batch_size, 7, 2, 20)
    

    # x_train, y_train = train_dg.__getitem__(0)
    # initial_lrate = 0.001
    # model = GetYoloV1Model(7, 2, 20, input_shape = (448, 448, 3))

    # pred = model(x_train)

    # loss = yolo_loss(tf.constant(y_train), pred)
    # print(loss)

if __name__ == "__main__":
    test()
