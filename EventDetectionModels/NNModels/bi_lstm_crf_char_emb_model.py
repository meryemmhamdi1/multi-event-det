import tensorflow as tf
from .model_utils import *
from DataModule.data_utils import *
from .base_model import *


class BiLSTMCRFChar(BaseModel):
    def __init__(self, args):
        super(BiLSTMCRFChar, self).__init__(args)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.args.vocab_tags.items()}

        self.tag_to_idx = {tag: idx for tag, idx in
                           self.args.vocab_tags.items()}

        ## Build Model
        self.define_placeholders()
        self.word_embeddings_ops()
        self.char_embeddings_ops()
        self.add_bi_lstm_logits_op()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op(args.optimizer, self.lr, self.loss,
                          args.clip)

        self.initialize_session()

    def define_placeholders(self):
        """Define variables entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                                       name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                                               name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                                       name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                                           name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                                     name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                                 name="lr")

    def word_embeddings_ops(self):
        ## Word Embeddings Layer: initialized with pre-trained embeddings if provided
        with tf.variable_scope("words"):
            if self.args.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                    name="_word_embeddings",
                    dtype=tf.float32,
                    shape=[self.args.nwords, self.args.dim_word])
            else:
                _word_embeddings = tf.Variable(
                    self.args.embeddings,
                    name="_word_embeddings",
                    dtype=tf.float32,
                    trainable=self.args.train_embed)

            print("self.args.nwords:", self.args.nwords)
            print("len(self.args.embeddings):", len(self.args.embeddings))
            self.word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                                                     self.word_ids, name="word_embeddings")

    def char_embeddings_ops(self):
        ## Character Embeddings Layer
        with tf.variable_scope("chars"):
            if self.args.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                    name="_char_embeddings",
                    dtype=tf.float32,
                    shape=[self.args.nchars, self.args.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                         self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                                             shape=[s[0]*s[1], s[-2], self.args.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.args.hidden_size_char,
                                                  state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.args.hidden_size_char,
                                                  state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, char_embeddings,
                    sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                                    shape=[s[0], s[1], 2*self.args.hidden_size_char])
                word_embeddings = tf.concat([self.word_embeddings, output], axis=-1)

        ## Concatenate word embeddings and character embeddings
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    def add_bi_lstm_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.args.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.args.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, self.word_embeddings,
                sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                                shape=[2*self.args.hidden_size_lstm, self.args.ntags])

            b = tf.get_variable("b", shape=[self.args.ntags],
                                dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.args.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.args.ntags])

    def add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With theCRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        if not self.args.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                                       tf.int32)

    def add_loss_op(self):
        """Defines the loss"""
        if self.args.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.labels, self.sequence_lengths)
            self.trans_params = trans_params # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)

    def run_epoch(self, train, dev, test, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.args.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        losses = []
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.args.learning_rate,
                                       self.args.dropout)

            _, train_loss, summary = self.sess.run(
                [self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])
            losses.append(train_loss)

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        metrics_train, summ_train = self.run_evaluate(train, "train")
        #print("metrics_train:", metrics_train)
        metrics_dev, summ_dev = self.run_evaluate(dev, "train")

        metrics_test, summ_test = self.run_evaluate(test, "test")

        self.file_writer.add_summary(summ_dev, epoch*nbatches + i)

        msg_train = "Training "+" - ".join(["{} {:04.2f}".format(k, v)
                                            for k, v in metrics_train.items() if k not in ["y_true", "y_pred"]])
        self.logger.info(msg_train)

        msg_dev = "Dev "+" - ".join(["{} {:04.2f}".format(k, v)
                                     for k, v in metrics_dev.items() if k not in ["y_true", "y_pred"]])
        self.logger.info(msg_dev)

        for lang in metrics_test:
            msg_test = "Testing on " + lang+" - ".join(["{} {:04.2f}".format(k, v)
                                              for k, v in metrics_test[lang].items() if k not in ["y_true", "y_pred"]])
            self.logger.info(msg_test)

        return metrics_train, metrics_dev, metrics_test, losses

    def run_evaluate(self, test, mode):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """

        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        y_true_all = []
        y_pred_all = []
        if mode == "train":
            # iterate over dataset
            for words, labels in minibatches(test, self.args.batch_size):
                labels_pred, sequence_lengths = self.predict_batch(words)

                y = []
                y_ = []
                for i in range(len(labels)):
                    y_s = []
                    for j in range(len(labels[i])):
                        y_s.append(self.idx_to_tag[labels[i][j]])
                    y.append(y_s)

                for i in range(len(labels_pred)):
                    y_s_ = []
                    for j in range(len(labels_pred[i])):
                        y_s_.append(self.idx_to_tag[labels_pred[i][j]])
                    y_.append(y_s_)

                y_true_all.append(y)
                y_pred_all.append(y_)
                for lab, lab_pred, length in zip(labels, labels_pred,
                                                 sequence_lengths):
                    lab = lab[:length]
                    lab_pred = lab_pred[:length]
                    accs += [a==b for (a, b) in zip(lab, lab_pred)]

                    lab_chunks = set(get_chunks(lab, self.args.vocab_tags))
                    lab_pred_chunks = set(get_chunks(lab_pred,
                                                     self.args.vocab_tags))

                    correct_preds += len(lab_chunks & lab_pred_chunks)
                    total_preds += len(lab_pred_chunks)
                    total_correct += len(lab_chunks)

            p = correct_preds / total_preds if correct_preds > 0 else 0
            r = correct_preds / total_correct if correct_preds > 0 else 0
            f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
            acc = np.mean(accs)

            summary = tf.Summary()
            summary.value.add(tag="acc", simple_value=acc)
            summary.value.add(tag="f1", simple_value=f1)
            summary.value.add(tag="p", simple_value=p)
            summary.value.add(tag="r", simple_value=r)

            return {"acc": 100*acc, "f1": 100*f1, "r": 100*r, "p": 100*p, "y_true": y_true_all, "y_pred": y_pred_all}\
                , summary

        else:
            metrics = {}
            summaries = {}
            for lang in test:
                for words, labels in minibatches_test(test[lang], self.args.batch_size):
                    labels_pred, sequence_lengths = self.predict_batch(words)

                    y = []
                    y_ = []
                    for i in range(len(labels)):
                        y_s = []
                        for j in range(len(labels[i])):
                            y_s.append(self.idx_to_tag[labels[i][j]])
                        y.append(y_s)

                    for i in range(len(labels_pred)):
                        y_s_ = []
                        for j in range(len(labels_pred[i])):
                            y_s_.append(self.idx_to_tag[labels_pred[i][j]])
                        y_.append(y_s_)

                    y_true_all.append(y)
                    y_pred_all.append(y_)
                    for lab, lab_pred, length in zip(labels, labels_pred,
                                                     sequence_lengths):
                        lab = lab[:length]
                        lab_pred = lab_pred[:length]
                        accs += [a==b for (a, b) in zip(lab, lab_pred)]

                        lab_chunks = set(get_chunks(lab, self.args.vocab_tags))
                        lab_pred_chunks = set(get_chunks(lab_pred,
                                                         self.args.vocab_tags))

                        correct_preds += len(lab_chunks & lab_pred_chunks)
                        total_preds += len(lab_pred_chunks)
                        total_correct += len(lab_chunks)

                p = correct_preds / total_preds if correct_preds > 0 else 0
                r = correct_preds / total_correct if correct_preds > 0 else 0
                f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
                acc = np.mean(accs)

                summary = tf.Summary()
                summary.value.add(tag="acc", simple_value=acc)
                summary.value.add(tag="f1", simple_value=f1)
                summary.value.add(tag="p", simple_value=p)
                summary.value.add(tag="r", simple_value=r)

                metrics.update({lang: {"acc": 100*acc, "f1": 100*f1, "r": 100*r, "p": 100*p, "y_true": y_true_all, "y_pred": y_pred_all}})
                summaries.update({lang: summary})

            return metrics, summaries

    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data
        if self.args.use_chars:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                                                   nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.args.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths

    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.args.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                [self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths


