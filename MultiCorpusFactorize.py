import tensorflow as tf
import math
import os
import nce_loss.nce_loss_with_factor_matrix as nce
import numpy as np

"""
Load the dictionary file to get (word,id) vocabulary
"""
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


def generate_batch(filename, vocabulary):
    f = open(filename)
    stream = []
    for line in f:
        words = line.strip().split(",")
        try:
            center_index = vocabulary[words[0]]
            context_index = vocabulary[words[1]]
        except KeyError:
            # Where we are looking at words that are not in the vocabulary because it is infrequent
            continue
        stream.append((center_index, context_index))
    f.close()
    return stream


"""
    Pipeline data with vocabulary and reversed vocabulary
"""
econs_stream_file = "/home/duc/Documents/homework/Research/Tensorflow/Codes/pipeline_data/econs_pipeline"
plos_stream_file = "/home/duc/Documents/homework/Research/Tensorflow/Codes/pipeline_data/plos_pipeline"
vocabulary = load_dictionary(os.path.join(os.getcwd(), "pipeline_data/vocabulary"))
reversed_vocabulary = {v: k for k, v in vocabulary.items()}

# Loading the data stream into an array
# Note that this will not do for larger corpora
econs_stream = generate_batch(econs_stream_file, vocabulary)
plos_stream = generate_batch(plos_stream_file, vocabulary)

# CONSTANTS DECLARATION
# TODO: toy example, to change for bigger corpus
# TODO: figure out how to perform reshaping of of the inputs work on matrix multiplication

vocabulary_size = 500
embedding_size = 20
batch_size = 32 # Size of 1 batch for stochastic gradient descent
num_negative_sample = 32 # Number of negative sample for each

"""
    Initialize tensorflow graph
"""
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
                                              contexts_vectors=train_context_A,
                                              center_word_embeddings=embed_A,
                                              num_sampled=num_negative_sample,
                                              num_classes=vocabulary_size,
                                              factor_matrix=matrix_factor_A)

    Y = tf.Variable(tf.random_uniform([embedding_size, embedding_size]))
    matrix_factor_B = tf.multiply(0.5, tf.add(Y, tf.transpose(Y)))
    loss_tensor_B = nce.nce_loss_multi_corpus(weights=nce_weights,
                                              biases=nce_biases,
                                              contexts_vectors=train_context_B,
                                              center_word_embeddings=embed_B,
                                              num_sampled=num_negative_sample,
                                              num_classes=vocabulary_size,
                                              factor_matrix=matrix_factor_B)

    # Stacked the loss tensor
    stacked_loss_tensor = tf.concat(values=[loss_tensor_A, loss_tensor_B], axis=0)
    nce_loss = tf.reduce_mean(stacked_loss_tensor)
    # nce_loss = tf.reduce_mean(loss_tensor_A)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
    optimization_ops = optimizer.minimize(nce_loss)

    # This is for validation and testing
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm

    init = tf.global_variables_initializer()


num_steps = 15000
# num_steps = 0
check_interval = 1000



with tf.Session(graph=graph) as session:
    init.run()
    average_loss = 0
    index_econs = 0
    econs_len = len(econs_stream)
    index_plos = 0
    plos_len = len(plos_stream)

    for step in range(num_steps):
        batch_inputs_A = np.ndarray(shape=(split_batch_size), dtype=np.int32)
        batch_context_A = np.ndarray(shape=(split_batch_size, 1), dtype=np.int32)

        batch_inputs_B = np.ndarray(shape=(split_batch_size), dtype=np.int32)
        batch_context_B = np.ndarray(shape=(split_batch_size, 1), dtype=np.int32)

        for i in range(split_batch_size):
            center, context = econs_stream[(index_econs + i) % econs_len]
            batch_inputs_A[i] = center
            batch_context_A[i,0] = context

            center, context = plos_stream[(index_plos + i) % plos_len]
            batch_inputs_B[i] = center
            batch_context_B[i,0] = context

        index_econs = (index_econs + split_batch_size) % econs_len
        index_plos = (index_plos + split_batch_size) % plos_len

        feed_dict = {train_inputs_A: batch_inputs_A,
                     train_context_A: batch_context_A,
                     train_inputs_B: batch_inputs_B,
                     train_context_B: batch_context_B}

        _, loss_nce = session.run([optimization_ops, nce_loss], feed_dict=feed_dict)

        if step % check_interval == 0:
            print("Step: ", step, " - loss: " , loss_nce)

    print("final loss: ", loss_nce)
    final_embeddings = normalized_embeddings.eval()

    Lambda_A, W_A, _ = tf.svd(matrix_factor_A)
    Lambda_B, W_B, _ = tf.svd(matrix_factor_B)

    Lambda_A = tf.nn.relu(Lambda_A)
    Lambda_B = tf.nn.relu(Lambda_B)

    Lambda_A = tf.diag(tf.sqrt(Lambda_A))
    Lambda_B = tf.diag(tf.sqrt(Lambda_B))

    modified_embeddings_A = tf.matmul(embeddings, tf.matmul(W_A, Lambda_A))
    modified_embeddings_B = tf.matmul(embeddings, tf.matmul(W_B, Lambda_B))

    validation_words = ["increase", "risk", "world", "potential", "degrees", "patterns"]
    validation_id = np.ndarray(shape=(len(validation_words)), dtype=np.int32)
    for i, word in enumerate(validation_words):  # get the ids of validation words
        validation_id[i] = vocabulary[word]

    valid_embedding_A = tf.nn.embedding_lookup(modified_embeddings_A, validation_id)
    valid_embedding_B = tf.nn.embedding_lookup(modified_embeddings_B, validation_id)

    similarity_A = tf.matmul(valid_embedding_A, modified_embeddings_A, transpose_b=True)
    similarity_B = tf.matmul(valid_embedding_B, modified_embeddings_B, transpose_b=True)
    sim_A = similarity_A.eval()
    sim_B = similarity_B.eval()

    top_k = 10  # 5 nearest words
    print()
    print("Validate WITH eigenvalue decomposition")
    for i, word in enumerate(validation_words):
        nearest_A = (-sim_A[i, :]).argsort()[1:top_k + 1]
        nearest_B = (-sim_B[i, :]).argsort()[1:top_k + 1]

        close_word_list_A = []
        for k in range(top_k):
            close_word = reversed_vocabulary[nearest_A[k]]
            close_word_list_A.append(close_word)

        close_word_list_B = []
        for k in range(top_k):
            close_word = reversed_vocabulary[nearest_B[k]]
            close_word_list_B.append(close_word)

        print("For word: ", word)
        print("Nearest neighbors in economic news corpus: ", close_word_list_A)
        print("Nearest neighbors in PLOS (ecology) corpus: ", close_word_list_B)
        print()

    for i, word in enumerate(validation_words):  # get the ids of validation words
        validation_id[i] = vocabulary[word]

    valid_embedding = tf.nn.embedding_lookup(normalized_embeddings, validation_id)
    similarity_A = tf.matmul(valid_embedding, tf.matmul(matrix_factor_A, normalized_embeddings, transpose_b=True))
    similarity_B = tf.matmul(valid_embedding, tf.matmul(matrix_factor_B, normalized_embeddings, transpose_b=True))

    sim_A = similarity_A.eval()
    sim_B = similarity_B.eval()

    top_k = 10  # 5 nearest words
    print()
    print("Validate without eigenvalue decomposition")
    for i, word in enumerate(validation_words):
        nearest_A = (-sim_A[i, :]).argsort()[1:top_k + 1]
        nearest_B = (-sim_B[i, :]).argsort()[1:top_k + 1]

        close_word_list_A = []
        for k in range(top_k):
            close_word = reversed_vocabulary[nearest_A[k]]
            close_word_list_A.append(close_word)

        close_word_list_B = []
        for k in range(top_k):
            close_word = reversed_vocabulary[nearest_B[k]]
            close_word_list_B.append(close_word)

        print("For word: ", word)
        print("Nearest neighbors in economic news corpus: ", close_word_list_A)
        print("Nearest neighbors in PLOS (ecology) corpus: ", close_word_list_B)
        print()


def validate_evd():
    with session.as_default():

        validation_words = ["increase", "bank", "risk", "world", "potential", "degrees", "patterns"]
        validation_id = np.ndarray(shape=(len(validation_words)), dtype=np.int32)
        for i,word in enumerate(validation_words): # get the ids of validation words
            validation_id[i] = vocabulary[word]

        valid_embedding = tf.nn.embedding_lookup(normalized_embeddings, validation_id)
        similarity_A = tf.matmul(valid_embedding, tf.matmul(matrix_factor_A, normalized_embeddings, transpose_b=True))
        similarity_B = tf.matmul(valid_embedding, tf.matmul(matrix_factor_B, normalized_embeddings, transpose_b=True))

        sim_A = similarity_A.eval()
        sim_B = similarity_B.eval()

        top_k = 10 # 5 nearest words
        print()
        print("Validate without eigenvalue decomposition")
        for i, word in enumerate(validation_words):
            nearest_A = (-sim_A[i,:]).argsort()[1:top_k+1]
            nearest_B = (-sim_B[i,:]).argsort()[1:top_k+1]

            close_word_list_A = []
            for k in range(top_k):
                close_word = reversed_vocabulary[nearest_A[k]]
                close_word_list_A.append(close_word)

            close_word_list_B = []
            for k in range(top_k):
                close_word = reversed_vocabulary[nearest_B[k]]
                close_word_list_B.append(close_word)

            print("For word: ", word)
            print("Nearest neighbors in economic news corpus: ", close_word_list_A)
            print("Nearest neighbors in PLOS (ecology) corpus: ", close_word_list_B)
            print()

        for i, word in enumerate(validation_words):  # get the ids of validation words
            validation_id[i] = vocabulary[word]

        valid_embedding = tf.nn.embedding_lookup(normalized_embeddings, validation_id)
        similarity_A = tf.matmul(valid_embedding, tf.matmul(matrix_factor_A, embeddings, transpose_b=True))
        similarity_B = tf.matmul(valid_embedding, tf.matmul(matrix_factor_B, embeddings, transpose_b=True))

        sim_A = similarity_A.eval()
        sim_B = similarity_B.eval()

        top_k = 10  # 5 nearest words
        print()
        print("Validate without eigenvalue decomposition")
        for i, word in enumerate(validation_words):
            nearest_A = (-sim_A[i, :]).argsort()[1:top_k + 1]
            nearest_B = (-sim_B[i, :]).argsort()[1:top_k + 1]

            close_word_list_A = []
            for k in range(top_k):
                close_word = reversed_vocabulary[nearest_A[k]]
                close_word_list_A.append(close_word)

            close_word_list_B = []
            for k in range(top_k):
                close_word = reversed_vocabulary[nearest_B[k]]
                close_word_list_B.append(close_word)

            print("For word: ", word)
            print("Nearest neighbors in economic news corpus: ", close_word_list_A)
            print("Nearest neighbors in PLOS (ecology) corpus: ", close_word_list_B)
            print()

"""
validate without finding evd
"""
def validate_without_evd():
    with session.as_default():
        # Find similar words in two corpuses
        validation_words = ["increase", "bank", "risk", "world", "potential", "degrees", "patterns"]
        validation_id = np.ndarray(shape=(len(validation_words)), dtype=np.int32)
        for i,word in enumerate(validation_words): # get the ids of validation words
            validation_id[i] = vocabulary[word]

        valid_embedding = tf.nn.embedding_lookup(embeddings, validation_id)
        similarity_A = tf.matmul(valid_embedding, tf.matmul(matrix_factor_A, embeddings, transpose_b=True))
        similarity_B = tf.matmul(valid_embedding, tf.matmul(matrix_factor_B, embeddings, transpose_b=True))

        sim_A = similarity_A.eval()
        sim_B = similarity_B.eval()

        top_k = 10 # 5 nearest words
        print()
        print("Validate without eigenvalue decomposition")
        for i, word in enumerate(validation_words):
            nearest_A = (-sim_A[i,:]).argsort()[1:top_k+1]
            nearest_B = (-sim_B[i,:]).argsort()[1:top_k+1]

            close_word_list_A = []
            for k in range(top_k):
                close_word = reversed_vocabulary[nearest_A[k]]
                close_word_list_A.append(close_word)

            close_word_list_B = []
            for k in range(top_k):
                close_word = reversed_vocabulary[nearest_B[k]]
                close_word_list_B.append(close_word)

            print("For word: ", word)
            print("Nearest neighbors in economic news corpus: ", close_word_list_A)
            print("Nearest neighbors in PLOS (ecology) corpus: ", close_word_list_B)
            print()

# validate_without_evd()
#
# validate_evd()