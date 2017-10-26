import tensorflow as tf
import math
import os
import nce_loss.nce_loss_with_factor_matrix as nce
import numpy as np


data_stream_1 = os.path.join(os.getcwd(), "pipeline_data/cnn_article")
data_stream_2 = os.path.join(os.getcwd(), "pipeline_data/breibart_article")

def load_dictionary(dictionary_file):
    f = open(dictionary_file, "r")
    vocabulary = {}
    for line in f:
        words = line.strip().split(",")
        if len(words) > 2:
            vocabulary[words[0]] = int(words[2])
        else:
            vocabulary[words[0]] = int(words[1])
    return vocabulary

vocabulary = load_dictionary(os.path.join(os.getcwd(), "pipeline_data/dictionary"))

# Loading the data stream into an array
# Note that this will not do for larger corpora

f = open(data_stream_1)
stream_1 = []
for line in f:
    words = line.strip().split(",")
    try:
        center_index = vocabulary[words[0]]
        context_index = vocabulary[words[1]]
    except KeyError:
        # Where we are looking at words that are not in the vocabulary because it is infrequent
        continue
    stream_1.append((center_index, context_index))
f.close()


f = open(data_stream_2)
stream_2 = []
for line in f:
    words = line.strip().split(",")
    try:
        center_index = vocabulary[words[0]]
        context_index = vocabulary[words[1]]
    except KeyError:
        # Where we are looking at words that are not in the vocabulary because it is infrequent
        continue
    stream_2.append((center_index, context_index))
f.close()

# CONSTANTS DECLARATION
# TODO: toy example, to change for bigger corpus
# TODO: figure out how to perform reshaping of of the inputs work on matrix multiplication

vocabulary_size = 300
embedding_size = 20
batch_size = 16 # Size of 1 batch for stochastic gradient descent
num_corpus = 2 # Number of corpus to consider so each batch will be divided equally by num_corpus
num_negative_sample = 10 # Number of negative sample for each

graph = tf.Graph() # tensor flow graph -> how computations are transferred from nodes to nodes

with graph.as_default():
    # train_context and train_inputs (basically two arrays of int (the id for the word)
    # Separate this by 2
    split_batch_size = int(batch_size/2)

    train_inputs_A = tf.placeholder(tf.int32, shape=[split_batch_size])
    train_inputs_B = tf.placeholder(tf.int32, shape=[split_batch_size])

    train_context_A = tf.placeholder(tf.int32, shape=[split_batch_size, 1])
    train_context_B = tf.placeholder(tf.int32, shape=[split_batch_size, 1])

    # This matrix embeddings is the final product that we want for the word embedding
    # This is the common word embedding matrix that all the corpora will share in the end
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

    embed_A = tf.nn.embedding_lookup(embeddings, train_inputs_A)
    embed_B = tf.nn.embedding_lookup(embeddings, train_inputs_B)

    # TODO: modify this part so that we have 2 different sets of nce_weights/nce_biases
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                  stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # todo: matrix_factor_A is symmetric because we implicitly enforce that
    X = tf.Variable(tf.random_uniform([embedding_size, embedding_size]))
    matrix_factor_A = tf.multiply(0.5, tf.add(X, tf.transpose(X)))
    loss_tensor_A = nce.nce_loss_multi_corpus(weights=nce_weights,
                                              biases=nce_biases,
                                              contexts=train_context_A,
                                              center_word_embeddings=embed_A,
                                              num_sampled=num_negative_sample,
                                              num_classes=vocabulary_size,
                                              factor_matrix=matrix_factor_A)

    Y = tf.Variable(tf.random_uniform([embedding_size, embedding_size]))
    matrix_factor_B = tf.multiply(0.5, tf.add(Y, tf.transpose(Y)))
    loss_tensor_B = nce.nce_loss_multi_corpus(weights=nce_weights,
                                              biases=nce_biases,
                                              contexts=train_context_B,
                                              center_word_embeddings=embed_B,
                                              num_sampled=num_negative_sample,
                                              num_classes=vocabulary_size,
                                              factor_matrix=matrix_factor_B)

    # Stacked the loss tensor
    stacked_loss_tensor = tf.concat(values=[loss_tensor_A, loss_tensor_B], axis=0)
    nce_loss = tf.reduce_mean(stacked_loss_tensor)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
    optimization_ops = optimizer.minimize(nce_loss)

    init = tf.global_variables_initializer()


num_steps = 10000
check_step = 1000

with tf.Session(graph=graph) as session:
    init.run()
    average_loss = 0
    index_1 = 0
    index_2 = 0

    for step in range(num_steps):
        batch_inputs_A = np.ndarray(shape=(split_batch_size), dtype=np.int32)
        batch_context_A = np.ndarray(shape=(split_batch_size, 1), dtype=np.int32)

        batch_inputs_B = np.ndarray(shape=(split_batch_size), dtype=np.int32)
        batch_context_B = np.ndarray(shape=(split_batch_size, 1), dtype=np.int32)

        for i in range(split_batch_size):
            center, context =  stream_1[(index_1 + i)%len(stream_1)]
            batch_inputs_A[i] = center
            batch_context_A[i] = context
            index_1 = (index_1 + split_batch_size)%len(stream_1)

            center, context =  stream_2[(index_2 + i)%len(stream_2)]
            batch_inputs_B[i] = center
            batch_context_B[i] = context
            index_2 = (index_2 + split_batch_size)%len(stream_2)

        feed_dict = {train_inputs_A: batch_inputs_A,
                     train_context_A: batch_context_A,
                     train_inputs_B: batch_inputs_B,
                     train_context_B: batch_context_B}

        _, loss_nce = session.run([optimization_ops, nce_loss], feed_dict=feed_dict)

        if step % check_step == 0:
            print(loss_nce)

    # print("factor A")
    # print(matrix_factor_A.eval())
    #
    # print("factor B")
    # print(matrix_factor_B.eval())



