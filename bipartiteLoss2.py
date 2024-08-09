from scipy.optimize import linear_sum_assignment
import tensorflow as tf
import keras
from keras import layers
from keras import Input
import numpy as np

def hungarian_algorithm(cost_matrix):

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    return row_ind.astype(np.int64), col_ind.astype(np.int64)

def bipartite_loss(coords1, coords2):

    coords2 = coords2.to_tensor()

    coords1 = tf.cast(coords1, tf.float32)
    coords2 = tf.cast(coords2, tf.float32)

    coords1_expanded = tf.expand_dims(coords1, axis=1)  # shape: (num_points1, 1, 2)
    coords2_expanded = tf.expand_dims(coords2, axis=0)  # shape: (1, num_points2, 2)

    cost_matrix = tf.reduce_sum(tf.square(coords1_expanded - coords2_expanded), axis=-1)

    row_ind, col_ind = tf.py_function(hungarian_algorithm, [cost_matrix], [tf.int64, tf.int64])
    indices = tf.stack([row_ind, col_ind], axis=1)

    total_sum = tf.reduce_mean(tf.gather_nd(cost_matrix, indices))

    return total_sum

@tf.function
def loss_func(y_pred, y_true):

    y_pred_reshaped = tf.map_fn(lambda pred: tf.reshape(pred, [-1, 2]), y_pred)
    loss_per_example = tf.map_fn(
        lambda p_t: bipartite_loss(p_t[0], p_t[1]),
        (y_pred_reshaped, y_true),
        fn_output_signature=tf.float32
    )


    return tf.reduce_mean(loss_per_example)