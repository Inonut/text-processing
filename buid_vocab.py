from config import Config
from util.data_util import CoNLLDataset, UNK, NUM, \
    export_glove_train, lowercase_processing
from util.general_util import Logger


def main():
    # get config and processing of words
    config = Config()
    logger = Logger.get_instance()

    # Generators
    train = CoNLLDataset(config.filename_train)  # train data
    glove = CoNLLDataset(config.filename_glove) if config.use_pretrained else None

    logger.info("Building vocab...")
    vocab_words, vocab_tags, vocab_chars = train.make_vocab()
    vocab_words = set(map(lambda word: lowercase_processing(word), vocab_words))
    logger.info("- done. {} word tokens".format(len(vocab_words)))
    logger.info("- done. {} tag tokens".format(len(vocab_tags)))
    logger.info("- done. {} char tokens".format(len(vocab_chars)))

    # process real vocab
    if glove is not None:
        vocab_glove_words, _, _ = glove.make_vocab()
        logger.info("- done. {} glove word tokens".format(len(vocab_glove_words)))
        vocab_words = vocab_words & vocab_glove_words
    vocab_words.add(UNK)
    vocab_words.add(NUM)

    logger.info("Writing vocab...")
    train.write_vocab(vocab_words, config.filename_words)
    train.write_vocab(vocab_tags, config.filename_tags)
    train.write_vocab(vocab_chars, config.filename_chars)
    logger.info("- done. {} word tokens".format(len(vocab_words)))
    logger.info("- done. {} tag tokens".format(len(vocab_tags)))
    logger.info("- done. {} char tokens".format(len(vocab_chars)))

    if glove is not None:
        vocab = glove.load_vocab(config.filename_words)
        export_glove_train(vocab, config.filename_glove, config.filename_trimmed, config.dim_word)
        logger.info("- done. writing glove training")


if __name__ == "__main__":
    main()
