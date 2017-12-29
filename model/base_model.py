import os

import tensorflow as tf

from config import Config
from util.data_util import CoNLLDataset
from util.general_util import Logger


class BaseModel(object):

    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = Logger.get_instance()

    def build(self):
        self._add_placeholders()
        self._add_word_embeddings_op()
        self._add_char_embeddings_op()
        self._add_logits_op()
        self._add_prediction_op()
        self._add_loss_op()
        self.__add_train_op(self.config.lr_method, self.config.lr)
        self.__initialize_session()
        self.__add_summary()  # tensorboard

    def train(self, train: CoNLLDataset, dev: CoNLLDataset):
        best_score = 0
        n_epoch_no_improve = 0  # for early stopping

        for epoch in range(self.config.n_epochs):
            self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))

            score = self._run_epoch(train, dev, epoch)
            self.config.lr *= self.config.lr_decay  # decay learning rate

            if score >= best_score:
                n_epoch_no_improve = 0
                best_score = score
                self.save_session()
                self.logger.info("- new best score!")
            else:
                n_epoch_no_improve += 1
                if n_epoch_no_improve >= self.config.n_epoch_no_improve:
                    self.logger.info("- stop after {} epochs no improvement".format(n_epoch_no_improve))
                    break

    def evaluate(self, test):
        self.logger.info("Testing model over test set")
        metrics = self._run_evaluate(test)
        self.logger.info(" - ".join(["{} {:04.2f}".format(k, v) for k, v in metrics.items()]))

    def save_session(self):
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        self.saver.save(self.sess, self.config.dir_model)

    def close_session(self):
        self.sess.close()

    def restore_session(self, dir_model):
        self.logger.info("Reloading the latest trained model...")
        self.saver.restore(self.sess, dir_model)

    def reinitialize_weights(self, scope_name):
        variables = tf.contrib.framework.get_variables(scope_name)
        init = tf.variables_initializer(variables)
        self.sess.run(init)

    def __add_summary(self):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.dir_output, self.sess.graph)

    def __initialize_session(self):
        self.logger.info("Initializing tf session")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def __add_train_op(self, lr_method, lr):
        with tf.variable_scope("train_step"):
            optimizer = {
                'adam': tf.train.AdamOptimizer(lr),
                'adagrad': tf.train.AdagradOptimizer(lr),
                'gd': tf.train.GradientDescentOptimizer(lr),
                'rmsprop': tf.train.RMSPropOptimizer(lr)
            }[lr_method]

            self._add_train_op(optimizer)

    def _run_epoch(self, train: CoNLLDataset, dev: CoNLLDataset, epoch) -> int:
        raise NotImplementedError()

    def _run_evaluate(self, test: CoNLLDataset):
        raise NotImplementedError()

    def _add_placeholders(self):
        raise NotImplementedError()

    def _add_word_embeddings_op(self):
        raise NotImplementedError()

    def _add_char_embeddings_op(self):
        pass

    def _add_logits_op(self):
        raise NotImplementedError()

    def _add_prediction_op(self):
        pass

    def _add_loss_op(self):
        raise NotImplementedError()

    def _add_train_op(self, optimizer):
        raise NotImplementedError()
