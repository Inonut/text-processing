import numpy as np


class CoNLLDataset(object):

    def __init__(self, filename, processing_word=None, processing_tag=None):
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.length = None
        self.file_stream = None

    def __iter__(self):
        return self

    def __len__(self):
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length

    def __next__(self):
        if self.file_stream is None:
            self.rest_stream()
        return self.__next_sentence()

    def __next_sentence(self):
        words, tags = [], []
        while True:

            try:
                line = next(self.file_stream)
            except Exception as e:
                if len(words) != 0:
                    return words, tags

                self.rest_stream()
                raise e

            line = line.strip()

            if len(line) == 0 or line.startswith("-DOCSTART-"):
                if len(words) != 0:
                    return words, tags
            else:
                line_words = line.split(' ')

                words += [self.processing_word(line_words[0]) if self.processing_word is not None else line_words[0]]
                tags += [self.processing_tag(line_words[1]) if self.processing_tag is not None else line_words[1]]

    def rest_stream(self):
        if self.file_stream is not None:
            self.file_stream.close()
        self.file_stream = open(self.filename)

    def make_vocab(self):
        vocab_words = set()
        vocab_tags = set()
        vocab_chars = set()
        for words, tags in self:
            vocab_words.update(words)
            vocab_tags.update(tags)

        for word in vocab_words:
            vocab_chars.update(word)

        self.rest_stream()

        return vocab_words, vocab_tags, vocab_chars

    @staticmethod
    def load_vocab(filename):
        try:
            dictionary = dict()
            with open(filename) as f:
                for idx, word in enumerate(f):
                    word = word.strip()
                    dictionary[word] = idx

        except IOError:
            raise Exception("Unable to locate file {}".format(filename))
        return dictionary

    @staticmethod
    def write_vocab(vocab, filename):
        with open(filename, "w") as f:
            f.write('\n'.join(vocab))


UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"


def standard_processing(word):
    if word.isdigit():
        word = NUM

    return word


def lowercase_processing(word):
    word = standard_processing(word.lower())

    return word


def word_processing(vocab_words, vocab_chars):
    def processing(word):

        char_ids = []
        for char in word:
            # ignore chars out of vocabulary
            if char in vocab_chars:
                char_ids += [vocab_chars[char]]

        word = lowercase_processing(word)

        if word in vocab_words:
            word = vocab_words[word]
        else:
            word = vocab_words[UNK]

        return char_ids, word

    return processing


def tag_processing(vocab_tags):
    def processing(word):

        word = standard_processing(word)

        if word in vocab_tags:
            word = vocab_tags[word]
        else:
            word = vocab_tags[UNK]

        return word

    return processing


def export_glove_train(vocab, glove_filename, trimmed_filename, dim):
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def load_glove_train(trimmed_filename):
    try:
        with np.load(trimmed_filename) as data:
            return data["embeddings"]

    except IOError:
        raise Exception("Unable to locate file {}".format(trimmed_filename))


def minibatches(data, minibatch_size):
    words_batch, label_batch = [], []
    for (word, label) in data:
        if len(words_batch) == minibatch_size:
            yield words_batch, label_batch
            words_batch, label_batch = [], []

        words_batch += [zip(*word)]
        label_batch += [label]

    if len(words_batch) != 0:
        yield words_batch, label_batch


def fill_array(arr, max_length):
    return list(arr)[:max_length] + [0] * (max_length - len(arr))


def pad_sequences(sequences, sequences_length=None):
    sequences_length = max(
        map(lambda x: len(x) if type(x) is not int else 0, sequences)) if sequences_length is None else sequences_length

    seqs = []
    lengths = []
    for sequence in sequences:

        seq = fill_array(sequence if type(sequence) is not int else [sequence], sequences_length)
        length = len(sequence) if type(sequence) is not int else 0

        if type(sequence) is not int and type(sequence[0]) == list:
            max_length_word = max([max(map(lambda x: len(x), _seq)) for _seq in sequences])

            # seq = list(map(lambda x: [x] if type(x) == int else x, seq))
            seq, length = pad_sequences(seq, max_length_word)

        lengths += [length]
        seqs += [seq]

    return seqs, lengths


def get_chunks(seq, tags):
    chunks = []
    for i, label_id in enumerate(seq):
        if len(chunks) != 0 and chunks[-1][0] == label_id:
            chunks[-1][2] += 1
        else:
            chunks.append([label_id, i, i + 1])

    return list(map(lambda x: tuple(x), chunks))
