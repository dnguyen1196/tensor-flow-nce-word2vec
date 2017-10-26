import os
import collections
import csv
import string

# Load corpus file and build a dictionary on top of this
# Need to load large corpus
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
    # Assuming that all corpora are in corpora/ folder
    corpus_1 = os.path.join(os.getcwd(), "corpora/" + filename1)
    corpus_2 = os.path.join(os.getcwd(), "corpora/" + filename2)

    # Build key-count in dictionary
    key_count_pairs = build_common_dictionary(corpus_1, corpus_2, vocabulary_size)
    vocabulary = collections.defaultdict(int)

    # Save the result in pipe_line
    # how exactly does one handle word that does not appear in the dictionary?
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


def read_economics_news(input_file, output_file):
    f = open(output_file, "w")
    with open(input_file, "r") as infile:
        reader = csv.reader(infile)
        headers = next(reader)[1:]
        for row in reader:
            for key, value in zip(headers,row[1:]):
                if key == "headline" or key == "text":
                    f.write(value)
                    f.write("\n")
    f.close()


def read_plos_narrative(input_file, output_file):
    f = open(output_file, "w")
    with open(input_file, "r") as infile:
        reader = csv.reader(infile)
        headers = next(reader)[1:]
        for row in reader:
            for key, value in zip(headers,row[1:]):
                if key == "ab":
                    f.write(value)
                    f.write("\n")
    f.close()


# Extract texts from econs news corpus
# econ_news_corpus = "/home/duc/Documents/homework/Research/Tensorflow/Codes/corpora/Full-Economic-News-DFE-839861.csv"
# output_news_file = "/home/duc/Documents/homework/Research/Tensorflow/Codes/pipeline_data/economics-news-aggregated.txt"
# read_economics_news(econ_news_corpus, output_news_file)

# plos_narrative = "/home/duc/Documents/homework/Research/Tensorflow/Codes/corpora/PLOS_narrativity.csv"
# output_plos_file = "/home/duc/Documents/homework/Research/Tensorflow/Codes/pipeline_data/plos-narrative-aggregated.txt"
# read_plos_narrative(plos_narrative, output_plos_file)

def build_dictionary(corpus):
    f = open(corpus)
    corpus_dict = collections.defaultdict(int)
    for line in f:
        words = line.strip().split(" ")
        for word in words:
            cleaned = word.translate(string.maketrans("",""), string.punctuation).lower()
            corpus_dict[cleaned] += 1
    f.close()
    return corpus_dict


def build_dictionary_from_corpus(corpus_1, corpus_2, k, out_file_1, out_file_2):
    # build two dictionaries
    corpus_1 = build_dictionary(corpus_1)
    corpus_2 = build_dictionary(corpus_2)

    # Merge and pick the most frequent k words?
    # Merge the two corpus and pick words that are most common?
    pass


def build_pipeline(dictionary, corpus_1, corpus_2):
    pass

