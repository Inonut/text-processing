from util.data_util import lowercase_processing


class CoNLLDataset(object):

    def __init__(self, input_filename):
        self.filename = input_filename
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
                raise e

            line = line.strip()

            if len(line) == 0 or line.startswith("-DOCSTART-"):
                if len(words) != 0:
                    return words, tags
            else:
                line_words = line.split(' ')

                words += [line_words[0]]
                tags += [line_words[1]]

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


train = CoNLLDataset("data/test.txt")

print([next(train) for _ in range(2)])
print([next(train) for _ in range(2)])
