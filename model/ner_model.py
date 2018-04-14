from functools import reduce

import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector

from model.base_model import BaseModel
from util.data_util import CoNLLDataset, minibatches, pad_sequences, get_chunks, NONE
from util.general_util import Progbar


class NerModel(BaseModel):

    def _add_placeholders(self):
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, [None, None], "word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, [None], "sequence_lengths")

        # shape = (batch size, max length of sentence in batch, max length of word in sentence)
        self.char_ids = tf.placeholder(tf.int32, [None, None, None], "char_ids")

        # shape = (batch_size, max_length of sentence in batch)
        self.word_lengths = tf.placeholder(tf.int32, [None, None], "word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, [None, None], "labels")

        self.dropout = tf.placeholder(tf.float32, [], "dropout")
        self.lr = tf.placeholder(tf.float32, [], "lr")

    def _add_word_embeddings_op(self):

        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self._word_embeddings = tf.get_variable(
                    dtype=tf.float32,
                    shape=[self.config.n_words, self.config.dim_word],
                    name="_word_embeddings")
            else:
                self._word_embeddings = tf.Variable(
                    self.config.embeddings,
                    dtype=tf.float32,
                    trainable= False,
                    name="_word_embeddings")

            # shape (?, ?, self.config.dim_word)
            self.word_embeddings = tf.nn.embedding_lookup(self._word_embeddings, self.word_ids, name="word_embeddings")

            # for tensorboard
            tf.summary.histogram('_word_embeddings', self._word_embeddings)

    def _add_char_embeddings_op(self):
        with tf.variable_scope("chars"):
            # get char embeddings matrix
            _char_embeddings = tf.get_variable(
                dtype=tf.float32,
                shape=[self.config.n_chars, self.config.dim_char],
                name="_char_embeddings")

            # shape (?, ?, ?, self.config.dim_char)
            char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.char_ids, name="char_embeddings")

            # for tensorboard
            tf.summary.histogram('_char_embeddings', _char_embeddings)

            with tf.variable_scope("bi-lstm"):
                # make it to fit
                # because word lengths depends by bach_size; sentence can have different lengths
                s = tf.shape(char_embeddings)
                bach_size = s[0]
                max_length_sentence= s[1]
                mas_length_word = s[2]
                char_embeddings = tf.reshape(char_embeddings, [bach_size * max_length_sentence, mas_length_word, self.config.dim_char], "char_embeddings")
                word_lengths = tf.reshape(self.word_lengths, [bach_size * max_length_sentence], "word_lengths")

                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char_lstm, state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char_lstm, state_is_tuple=True)
                _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, char_embeddings, sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output, [bach_size, max_length_sentence, 2 * self.config.hidden_size_char_lstm], "output")

            word_embeddings = tf.concat([self.word_embeddings, output], axis=-1)
            self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    def _add_logits_op(self):

        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                        self.word_embeddings,
                                                                        sequence_length=self.sequence_lengths,
                                                                        dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32, shape=[2 * self.config.hidden_size_char_lstm, self.config.n_tags])

            b = tf.get_variable("b", shape=[self.config.n_tags], dtype=tf.float32, initializer=tf.zeros_initializer())

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2 * self.config.hidden_size_char_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, s[1], self.config.n_tags])

            # for tensorboard
            tf.summary.histogram('W', W)

    def _add_loss_op(self):
        # -log(x)
        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(self.logits,
                                                                         self.labels,
                                                                         self.sequence_lengths)
        self.trans_params = trans_params  # need to evaluate it for decoding
        self.loss = tf.reduce_mean(-log_likelihood)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)
        tf.summary.histogram("histogram loss", self.loss)

    def _add_train_op(self, optimizer):
        self.train_op = optimizer.minimize(self.loss)

    def _run_epoch(self, train: CoNLLDataset, dev: CoNLLDataset, epoch) -> int:
        n_batches = (len(train) + self.config.batch_size - 1) // self.config.batch_size
        prog = Progbar(target=n_batches)

        # iterate over dataset
        for i, (words, labels) in enumerate(minibatches(train, self.config.batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr, self.config.dropout)

            _, train_loss, summary = self.sess.run([self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * n_batches + i)

        metrics = self._run_evaluate(dev)
        self.logger.info(" - ".join(["{} {:04.2f}".format(k, v) for k, v in metrics.items()]))

        return metrics["f1"]

    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):

        char_ids, word_ids = zip(*words)

        word_ids, sequence_lengths = pad_sequences(word_ids)
        char_ids, word_lengths = pad_sequences(char_ids)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths,
            self.char_ids: char_ids,
            self.word_lengths: word_lengths
        }

        if labels is not None:
            labels, _ = pad_sequences(labels)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths

    def _run_evaluate(self, test: CoNLLDataset):

        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                cond = lambda x: x[0] != self.config.vocab_tags[NONE]   

                lab_chunks = set(filter(cond, get_chunks(lab, self.config.vocab_tags)))
                lab_pred_chunks = set(filter(cond, get_chunks(lab_pred, self.config.vocab_tags)))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return {"acc": 100 * acc, "f1": 100 * f1}

    def predict_batch(self, words):

        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        # get tag scores and transition params of CRF
        viterbi_sequences = []
        logits, trans_params = self.sess.run([self.logits, self.trans_params], feed_dict=fd)

        # iterate over the sentences because no batching in vitervi_decode
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit = logit[:sequence_length] # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
            viterbi_sequences += [viterbi_seq]

        return viterbi_sequences, sequence_lengths

    def predict(self, sentence):
        words = [self.config.processing_word(word) for word in sentence]
        labels_pred, _ = self.predict_batch([zip(*words)])

        idx_to_tag = {idx: tag for tag, idx in self.config.vocab_tags.items()}
        preds = [idx_to_tag[idx] for idx in list(labels_pred[0])]

        return preds

    def _add_word_embeddings_visualization(self):
        with tf.variable_scope("words"):
            final_embed_matrix = self.sess.run(self._word_embeddings)

        embedding_var = tf.Variable(final_embed_matrix, name='embedding')
        self.sess.run(embedding_var.initializer)

        # add embedding to the config file
        embedding = self.projector.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = '../' + self.config.filename_words

        # saves a configuration file that TensorBoard will read during startup.
        projector.visualize_embeddings(self.file_writer, self.projector)
        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(self.sess, self.config.path_ckpt, 1)
