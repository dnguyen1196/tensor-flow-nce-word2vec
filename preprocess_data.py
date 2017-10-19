import os
import sys
import collections

def load_text_file(filename):
    f = open(filename, "r")
    vocabulary = collections.defaultdict(int)
    for line in f:
        words = [word.lower() for word in line.strip().split()]
        for word in words:
            vocabulary[word] += 1
    f.close()
    return vocabulary


def build_common_dictionary(filename1, filename2, max_size):
    vocabulary_1 = load_text_file(filename1)
    vocabulary_2 = load_text_file(filename2)

    common_vocab = collections.defaultdict(int)

    for key in vocabulary_1:
        common_vocab[key] += vocabulary_1[key]
    for key in vocabulary_2:
        common_vocab[key] += vocabulary_2[key]

    key_count_pairs = [(key, common_vocab[key]) for key in common_vocab]
    key_count_pairs.sort(key= lambda x: -x[1])

    if len(key_count_pairs) < max_size:
        return key_count_pairs
    return key_count_pairs[:max_size]


def build_pipe_line(filename1, filename2, vocabulary_size, window_size):
    corpus_1 = os.path.join(os.getcwd(), "corpora/" + filename1)
    corpus_2 = os.path.join(os.getcwd(), "corpora/" + filename2)

    key_count_pairs = build_common_dictionary(corpus_1, corpus_2, vocabulary_size)
    vocabulary = collections.defaultdict(int)

    out = open(os.path.join(os.getcwd(), "pipeline_data/dictionary"), "w")
    for i,key_val in enumerate(key_count_pairs):
        key = key_val[0]
        val = key_val[1]
        vocabulary[key] = val
        s = key+","+str(i)+"\n"
        out.write(s)
    out.close()

    f = open(corpus_1, "r")
    word_array_1 = []
    for line in f:
        words = [word.lower() for word in line.strip().split()]
        word_array_1.extend(words)
    f.close()
    out = open(os.path.join(os.getcwd(), "pipeline_data/" + filename1), "w")
    for i in range(len(word_array_1)):
        center_word = word_array_1[i]
        for j in range(i - window_size, i + window_size + 1):
            if i == j:
                continue
            if j < 0 or j >= len(word_array_1):
                continue
            context = word_array_1[j]
            s = center_word + "," + context + "\n"
            out.write(s)
    out.close()

    f = open(corpus_2, "r")
    word_array_2 = []
    for line in f:
        words = [word.lower() for word in line.strip().split()]
        word_array_2.extend(words)
    f.close()
    out = open(os.path.join(os.getcwd(), "pipeline_data/" + filename2), "w")
    for i in range(len(word_array_2)):
        center_word = word_array_2[i]
        for j in range(i - window_size, i + window_size + 1):
            if i == j:
                continue
            if j < 0 or j >= len(word_array_2):
                continue
            context = word_array_1[j]
            if context not in vocabulary:
                continue
            s = center_word + "," + context + "\n"
            out.write(s)
    out.close()

build_pipe_line("cnn_article", "breibart_article", 300, 2)




