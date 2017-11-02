
# generate batch
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
