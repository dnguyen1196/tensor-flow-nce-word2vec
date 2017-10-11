import datetime as dt
import math
import os
import random
import zipfile
import tensorflow as tf
import collections
import numpy as np

from custom_optimizer.CustomOptimizer import CustomOptimizer


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(words, n_words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def collect_data(vocabulary_size=10000):
    filename = os.path.join(os.getcwd(), "corpora/text8.zip")
    vocabulary = read_data(filename)
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)
    del vocabulary
    return data, count, dictionary, reverse_dictionary


def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window input_word skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # input word at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]  # this is the input word
            context[i * num_skips + j, 0] = buffer[target]  # these are the context words
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, context


#################################################################################
#
#
#                       END OF FUNCTION DECLARATIONS
#
#
##################################################################################


data_index = 0
vocabulary_size = 10000
data, count, dictionary, reverse_dictionary = collect_data(vocabulary_size=vocabulary_size)

batch_size = 128
embedding_size = 300  # Dimension of the embedding vector.
skip_window = 2       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.


# VALIDATION PARAMETERS
# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph() # tensor flow graph -> how computations are transferred from nodes to nodes
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size]) # This is a place holder that will get
    # value of the id of the words
    train_context = tf.placeholder(tf.int32, shape=[batch_size, 1])
    # This is the context (surrounding words of the input
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    # This is a built in tensorflow constructs for quick word representation look up
    # Initialize the weights and biases variables, remember, Variables are the things
    # that get changed when tensorflow runs
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # TODO: implement my own loss tensor function
    # This should combines the calls that calculate the nce_loss function
    loss_tensor = tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_context,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size)\

    nce_loss = tf.reduce_mean(loss_tensor)

    # This should not change
    optimizer = CustomOptimizer(1.0)
    grads_and_vars = optimizer.compute_gradients(nce_loss)
    opt_operation = optimizer.apply_gradients(grads_and_vars)

    init = tf.global_variables_initializer()


num_steps = 10000
step_size = 1000
nce_start_time = dt.datetime.now()

with tf.Session(graph=graph) as session:
    init.run()
    average_loss = 0

    for step in range(num_steps):
        batch_inputs, batch_context = generate_batch(data, batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_context: batch_context}

        _, loss_val = session.run([opt_operation, nce_loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % step_size == 0:
            if step > 0:
                average_loss /= step_size
            print('Average loss at step ' + str(step) + ' : ' + str(average_loss))
            average_loss = 0


nce_end_time = dt.datetime.now()
print(("NCE method took {} seconds to run " +  str(num_steps) + " iterations").format((nce_end_time-nce_start_time).total_seconds()))