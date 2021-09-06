import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, concatenate


def xywh2minmax(xy, wh):
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2

    return xy_min, xy_max


def iou(pred_mins, pred_maxes, true_mins, true_maxes):

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_wh = pred_maxes - pred_mins
    true_wh = true_maxes - true_mins
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    return iou_scores


def yolo_head(feats):
    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = tf.shape(feats)[1:3]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = tf.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = tf.tile(
        tf.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(tf.transpose(conv_width_index))
    conv_index = tf.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = tf.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = tf.cast(conv_index, K.dtype(feats))

    conv_dims = tf.cast(tf.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

    box_xy = (feats[..., :2] + conv_index) / conv_dims * 448
    box_wh = feats[..., 2:4] * 448

    return box_xy, box_wh


def yolo_loss1(y_true, y_pred):

    label_class = y_true[..., :20]  # ? * 7 * 7 * 20
    label_box = y_true[..., 21:25]  # ? * 7 * 7 * 4
    response_mask = y_true[..., 20]  # ? * 7 * 7
    response_mask = tf.expand_dims(response_mask, axis=-1)  # ? * 7 * 7 * 1

    predict_class = y_pred[..., :20]  # ? * 7 * 7 * 20
    predict_trust = concatenate([ y_pred[..., 20:21], y_pred[..., 25:26] ])  # ? * 7 * 7 * 2
    predict_box = concatenate([ y_pred[..., 21:25], y_pred[..., 26:] ])  # ? * 7 * 7 * 8

    

    _label_box = tf.reshape(label_box, [-1, 7, 7, 1, 4])
    _predict_box = tf.reshape(predict_box, [-1, 7, 7, 2, 4])

    label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    label_xy = tf.expand_dims(label_xy, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_wh = tf.expand_dims(label_wh, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_xy_min, label_xy_max = xywh2minmax(label_xy, label_wh)  # ? * 7 * 7 * 1 * 1 * 2, ? * 7 * 7 * 1 * 1 * 2

    predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2
    predict_xy = K.expand_dims(predict_xy, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_wh = K.expand_dims(predict_wh, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_xy_min, predict_xy_max = xywh2minmax(predict_xy, predict_wh)  # ? * 7 * 7 * 2 * 1 * 2, ? * 7 * 7 * 2 * 1 * 2

    iou_scores = iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # ? * 7 * 7 * 2 * 1
    best_ious = K.max(iou_scores, axis=4)  # ? * 7 * 7 * 2
    best_box = K.max(best_ious, axis=3, keepdims=True)  # ? * 7 * 7 * 1

    box_mask = tf.cast(best_ious >= best_box, K.dtype(best_ious))  # ? * 7 * 7 * 2

    no_object_loss = 0.5 * (1 - box_mask * response_mask) * tf.square(0 - predict_trust)
    object_loss = box_mask * response_mask * tf.square(1 - predict_trust)
    confidence_loss = no_object_loss + object_loss
    confidence_loss = tf.reduce_sum(confidence_loss)

    class_loss = response_mask * tf.square(label_class - predict_class)
    class_loss = tf.reduce_sum(class_loss)

    _label_box = tf.reshape(label_box, [-1, 7, 7, 1, 4])
    _predict_box = tf.reshape(predict_box, [-1, 7, 7, 2, 4])

    label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2

    box_mask = tf.expand_dims(box_mask, axis=-1)
    response_mask = tf.expand_dims(response_mask, axis=-1)

    box_loss = 5 * box_mask * response_mask * tf.square((label_xy - predict_xy) / 448.)
    box_loss += 5 * box_mask * response_mask * tf.square((tf.sqrt(label_wh) - tf.sqrt(predict_wh)) / 448.)#448
    box_loss = tf.reduce_sum(box_loss)

    loss = confidence_loss + class_loss + box_loss

    return loss






def test():

    tf.random.set_seed(5)
    x = tf.random.normal([4,7,7,30], 0, 1, tf.float32)
    y = tf.random.normal([4,7,7,30], 0, 1, tf.float32)

    loss = yolo_loss1(x,y)
    print(loss)
    x = 1 


if __name__ == "__main__":
    test()
