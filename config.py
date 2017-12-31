import os

from util.data_util import load_glove_train, word_processing, tag_processing, CoNLLDataset
from util.general_util import Logger


class Config(object):

    def __init__(self) -> None:
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        Logger(self.path_log)

    def load_external_config(self):
        # Vocabulary
        self.vocab_words = CoNLLDataset.load_vocab(self.filename_words)
        self.vocab_tags = CoNLLDataset.load_vocab(self.filename_tags)
        self.vocab_chars = CoNLLDataset.load_vocab(self.filename_chars)

        self.n_words = len(self.vocab_words)
        self.n_chars = len(self.vocab_chars)
        self.n_tags = len(self.vocab_tags)

        # Processing functions
        self.processing_word = word_processing(self.vocab_words, self.vocab_chars)
        self.processing_tag = tag_processing(self.vocab_tags)

        # Pre-trained embeddings
        self.embeddings = load_glove_train(self.filename_trimmed) if self.use_pretrained else None

        return self

    # general config
    dir_output = "results/"
    dir_model = dir_output + "model.weights/"
    path_ckpt = dir_output + "model3.ckpt"
    path_log = dir_output + "log.txt"

    # dataset
    filename_dev = "data/coNLL/eng/eng.testa"
    filename_test = "data/coNLL/eng/eng.testb"
    filename_train = "data/coNLL/eng/eng.train"

    # vocab (created from dataset with build_data.py)
    filename_words = "data/words.txt"
    filename_tags = "data/tags.txt"
    filename_chars = "data/chars.txt"

    # embeddings
    dim_word = 300
    dim_char = 100

    # glove files
    filename_glove = "data/glove.6B/glove.6B.{}d.txt".format(dim_word)
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed = "data/glove.6B.{}d.trimmed.npz".format(dim_word)
    use_pretrained = True

    # training
    # train_embeddings = False
    n_epochs = 5
    dropout = 0.5  # rate to randomize
    batch_size = 20  # nb of sentence
    lr_method = "adam"
    lr = 0.001  # learn rate
    lr_decay = 0.9  # adjust learn rete
    n_epoch_no_improve = 3

    hidden_size_char = 100  # lstm on chars
    hidden_size_lstm = 300  # lstm on word embeddings
