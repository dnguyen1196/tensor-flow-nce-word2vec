import collections
import datetime as dt
import math
import os
import random
import zipfile

import numpy as np
import tensorflow as tf

from custom_optimizer.CustomOptimizer import CustomOptimizer
from custom_optimizer.AdaptedOptimizer import AdaptedOptimizer

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


def pretty_print_gradient(gradient):
    print("Gradient information: ")
    print(gradient.name)


def pretty_print_variable(variable):
    print("Variable information: "),
    print(variable.initial_value)
    print(variable.name)
    print(variable.value())


def gradient_processing(grads_and_var):
    for (gradient, variable) in grads_and_var:
        if gradient:
            pretty_print_gradient(gradient)
        if variable:
            pretty_print_variable(variable)
    return grads_and_var


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

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_context = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)


    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)



    # Construct the variables for the softmax
    weights = tf.Variable(tf.truncated_normal([embedding_size, vocabulary_size],stddev=1.0 / math.sqrt(embedding_size)))
    biases = tf.Variable(tf.zeros([vocabulary_size]))
    hidden_out = tf.transpose(tf.matmul(tf.transpose(weights), tf.transpose(embed))) + biases


    train_one_hot = tf.one_hot(train_context, vocabulary_size)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out, labels=train_one_hot))

    # convert train_context to a one-hot format
    train_one_hot = tf.one_hot(train_context, vocabulary_size)

    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    loss_tensor = tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_context,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size)\

    print("loss_tensor:...", loss_tensor)
    nce_loss = tf.reduce_mean(loss_tensor)

    optimizer = CustomOptimizer(1.0)
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    # TODO: figure out how compute gradients return a list of tuples
    # And can we replace this with a different function?
    grads_and_vars = optimizer.compute_gradients(nce_loss)
    opt_operation = optimizer.apply_gradients(gradient_processing(grads_and_vars))

    # print ("Loss function class: ", nce_loss.__class__)
    # print ("Optimizer class: ", optimizer.__class__)
    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()


num_steps = 10000
step_size = 1000
nce_start_time = dt.datetime.now()

with tf.Session(graph=graph) as session:
    init.run()
    average_loss = 0
    # optimizer = CustomOptimizer(1.0)
    # grads_and_vars = optimizer.compute_gradients(nce_loss)

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


    sim = similarity.eval()
    for i in range(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        close_word_list = []
        for k in range(top_k):
            close_word = reverse_dictionary[nearest[k]]
            close_word_list.append(close_word)

        print (log_str + ", ".join(close_word_list))

    final_embeddings = normalized_embeddings.eval()
    print (final_embeddings[0])


nce_end_time = dt.datetime.now()
print(("NCE method took {} seconds to run " +  str(num_steps) + " iterations").format((nce_end_time-nce_start_time).total_seconds()))