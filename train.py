from config import Config
from model.ner_model import NerModel
from util.data_util import CoNLLDataset


def main():
    # create instance of config
    config = Config().load_external_config()

    # build model
    model = NerModel(config)
    model.build()
    # model.restore_session("results/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets
    dev = CoNLLDataset(config.filename_dev, config.processing_word, config.processing_tag)
    train = CoNLLDataset(config.filename_train, config.processing_word, config.processing_tag)

    # train model
    model.train(train, dev)


if __name__ == "__main__":
    main()
