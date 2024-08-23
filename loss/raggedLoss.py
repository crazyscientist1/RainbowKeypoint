import tensorflow as tf
from scipy.optimize import linear_sum_assignment
import numpy as np


def hungarian_algorithm(cost_matrix):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind.astype(np.int32), col_ind.astype(np.int32)

def bipartite_loss(y_pred, y_true):
    print("sdfa")
    y_true = y_true.to_tensor()

    # y_pred = tf.transpose(y_pred, (2, 0, 1))

    print(y_pred.shape)
    print(y_true)
    print(y_true.shape)

    y_pred = tf.cast(y_pred, dtype=tf.float32)
    y_true = tf.cast(y_true, dtype=tf.float32)

    cost_matrix = tf.map_fn(
        lambda i: tf.map_fn(
            lambda j: tf.reduce_sum(tf.square(y_pred[i] - y_true[j])),
            tf.range(y_true.shape[0]),
            fn_output_signature=tf.float32
        ),
        tf.range(y_pred.shape[0]),
        fn_output_signature=tf.float32
    )

    row_ind, col_ind = tf.py_function(hungarian_algorithm, [cost_matrix], [tf.int64, tf.int64])

    indices = tf.stack([row_ind, col_ind], axis=1)

    total_sum = tf.reduce_mean(tf.gather_nd(cost_matrix, indices))

    return total_sum

@tf.function
def loss_func(y_true, y_pred):
    y_pred = tf.transpose(y_pred, (0, 3, 1, 2))

    print(y_pred.shape)
    
    loss_per_example = tf.map_fn(lambda i: bipartite_loss(y_pred[i], y_true[i]), tf.range(tf.shape(y_pred)[0]),
                                 fn_output_signature=tf.float32)

    return tf.reduce_mean(loss_per_example)