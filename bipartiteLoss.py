from scipy.optimize import linear_sum_assignment
import tensorflow as tf
import keras
from keras import layers
from keras import Input
import numpy as np
from itertools import product

def hungarian_algorithm(cost_matrix):

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    return row_ind.astype(np.int64), col_ind.astype(np.int64)


def bipartite_loss(yPred, yTrue):
    costMatrix = np.zeros((len(yPred), len(yTrue)))
    for x, y in product(range((len(yPred))), range(len(yTrue))):
        print(x,y)
        costMatrix[x,y] = np.sum(np.square(yPred[x]-yTrue[y]))

    row_ind, col_ind = tf.py_function(hungarian_algorithm, [costMatrix], [tf.int64, tf.int64])
    indices = tf.stack([row_ind, col_ind], axis=1)

    total_sum = tf.reduce_mean(tf.gather_nd(costMatrix, indices))

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
