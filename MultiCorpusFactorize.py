import tensorflow as tf
import math
import nce_loss.nce_helper_funcs as nce
from custom_optimizer.CustomOptimizer import CustomOptimizer

"""
Suppose we only consider 2 corpus
"""


# TODO: toy example, to change for bigger corpus
vocabulary_size = 10000
embedding_size = 100

batch_size = 128 # Size of 1 batch for stochastic gradient descent

num_negative_sample = 64

graph = tf.Graph() # tensor flow graph -> how computations are transferred from nodes to nodes
with graph.as_default():
    # train_context and train_inputs (basically two arrays of int (the id for the word)
    train_context = tf.placeholder(tf.int32, shape=[batch_size,1])
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])

    # This matrix embeddings is the final product that we want for the word embedding
    # This is the common word embedding matrix that all the corpora will share in the end
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)


    # TODO: modify this part so that we have 2 different sets of nce_weights/nce_biases
    # What exactly do we need the nce_weights for in the end we want the embeddings?
    # Different corpus will have this different nce_weights?
    # What exactly is this where does this come from?
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                  stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    X = tf.Variable(tf.random_uniform([embedding_size, embedding_size]))
    matrix_factor_B = tf.multiply(0.5, tf.add(X, tf.transpose(X)))

    loss_tensor_B = nce.nce_loss_special(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_context,
                       inputs=embed,
                       num_sampled=num_negative_sample,
                       num_classes=vocabulary_size,
                       factor_matrix=matrix_factor_B)

    X = tf.Variable(tf.random_uniform([embedding_size, embedding_size]))
    matrix_factor_A = tf.multiply(0.5, tf.add(X, tf.transpose(X)))

    loss_tensor_A = nce.nce_loss_special(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_context,
                       inputs=embed,
                       num_sampled=num_negative_sample,
                       num_classes=vocabulary_size,
                       factor_matrix=matrix_factor_A)

    stacked_loss_tensor = tf.concat(values=[loss_tensor_A, loss_tensor_B], axis=0)
    print("stacked_loss_tensor: ", stacked_loss_tensor)

    nce_loss = tf.reduce_mean(stacked_loss_tensor)

    optimizer = CustomOptimizer(learning_rate=1.0)

    # Think of this as the engine of the gradient descent
    # This does automatic differentiation and apply the gradient on the variables
    # in the graph
    optimization_ops = optimizer.minimize(nce_loss)

