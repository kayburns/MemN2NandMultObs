"""End-To-End Memory Networks.

The implementation is based on http://arxiv.org/abs/1503.08895 [1]
"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from six.moves import range


from utils import position_encoding


def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.name_scope(name, "add_gradient_noise", [t, stddev]) as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)


def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.name_scope(name, "zero_nil_slot", [t]) as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.pack([1, s]))
        return tf.concat(0, [z, tf.slice(t, [1, 0], [-1, -1])], name=name)


class MemN2N(object):
    """End-To-End Memory Network."""

    def __init__(self, 
                 batch_size,    
                 vocab_size,    
                 sentence_size,     
                 memory_size,   
                 num_caches,   
                 embedding_size,
                 vocab_dict,
                 reverse_vocab_dict,
                 hops=3,
                 max_grad_norm=40.0,
                 nonlin=None,
                 share_type='adjacent',
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 optimizer=tf.train.AdamOptimizer(learning_rate=1e-2),
                 encoding=position_encoding,
                 temporal_encoding=True,
                 session=tf.Session(),
                 name='MemN2N'
                ):
        """Creates an End-To-End Memory Network

        Args:
            batch_size: The size of the batch.

            vocab_size: The size of the vocabulary (should include the nil word). The nil word
            one-hot encoding should be 0.

            sentence_size: The max size of a sentence in the data. All sentences should be padded
            to this length. If padding is required it should be done with nil one-hot encoding (0).

            memory_size: The max size of the memory. Since Tensorflow currently does not support jagged arrays
            all memories must be padded to this length. If padding is required, the extra memories should be
            empty memories; memories filled with the nil word ([0, 0, 0, ......, 0]).

            embedding_size: The size of the word embedding.

            hops: The number of hops. A hop consists of reading and addressing a memory slot.
            Defaults to `3`.

            max_grad_norm: Maximum L2 norm clipping value. Defaults to `40.0`.

            nonlin: Non-linearity. Defaults to `None`.

            initializer: Weight initializer. Defaults to `tf.random_normal_initializer(stddev=0.1)`.

            optimizer: Optimizer algorithm used for SGD. Defaults to `tf.train.AdamOptimizer(learning_rate=1e-2)`.

            encoding: A function returning a 2D Tensor (sentence_size, embedding_size). Defaults to `position_encoding`.

            session: Tensorflow Session the model is run with. Defaults to `tf.Session()`.

            name: Name of the End-To-End Memory Network. Defaults to `MemN2N`.
        """

        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._sentence_size = sentence_size
        self._memory_size = memory_size
        self._num_caches = num_caches
        self._embedding_size = embedding_size
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._init = initializer
        self._opt = optimizer
        self._name = name

        self._vocab_dict = vocab_dict
        self._reverse_vocab_dict = reverse_vocab_dict

        self._build_inputs()
        self._build_vars(share_type)
        self._encoding = tf.constant(encoding(self._sentence_size, self._embedding_size), name="encoding")

        # cross entropy
        logits = self._inference(self._stories, self._observers, self._queries, share_type) # (batch_size, vocab_size)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, tf.cast(self._answers, tf.float32), name="cross_entropy")
        cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")

        # loss op
        self.loss_op = loss_op = cross_entropy_sum

        # gradient pipeline
        grads_and_vars = self._opt.compute_gradients(loss_op)
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in grads_and_vars]
        grads_and_vars = [(add_gradient_noise(g), v) for g, v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        self.train_op = train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")

        # predict ops
        self.predict_op = predict_op = tf.argmax(logits, 1, name="predict_op")
        self.predict_proba_op = predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")
        self.predict_log_proba_op = predict_log_proba_op = tf.log(predict_proba_op, name="predict_log_proba_op")

        # Initialize the graph
        init_op = tf.initialize_all_variables()
        self._sess = session
        self._sess.run(init_op)

    def _build_inputs(self):
        self._stories = tf.placeholder(tf.int32, [None, self._memory_size, self._sentence_size], name="stories")
        self._observers = tf.placeholder(tf.int32, [None, self._memory_size, self._num_caches], name="stories")
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None, self._vocab_size], name="answers")

    def _build_vars(self, share_type):
        with tf.variable_scope(self._name):

            nil_word_slot = tf.zeros([1, self._embedding_size])

            self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")
            self.W = tf.Variable(self._init([self._embedding_size, self._vocab_size]), name="W")

            self.A = []
            self.C = []

            self.TA = []
            self.TC = []

            B = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size - 1, self._embedding_size]) ])
            self.B = tf.Variable(B, name="B")

            if share_type == 'adjacent':

                self.A += [self.B]
                self.TA += [tf.zeros([self._memory_size, self._embedding_size])]  # unclear in the paper to what this matrix is tied

                C = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size - 1, self._embedding_size]) ])
                self.C += [tf.Variable(C, name="C_0")]

                TC = self._init([self._memory_size, self._embedding_size])
                self.TC += [tf.Variable(TC, name='TC_0')]

                for i in range(1, self._hops - 1):

                    self.A += [self.C[-1]]
                    self.TA += [self.TC[-1]]

                    C = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size - 1, self._embedding_size]) ])
                    self.C += [tf.Variable(C, name="C_%d" % i)]

                    TC = self._init([self._memory_size, self._embedding_size])
                    self.TC += [tf.Variable(TC, name='TC_%d' % i)]

                if self._hops > 1:
                    self.A += [self.C[-1]]
                    self.TA += [self.TC[-1]]

                    self.C += [tf.transpose(self.W, [1, 0])]
                    #self.TC += [tf.Variable(TC), name='TC_%d' % i)]  
                    self.TC += [tf.zeros([self._memory_size, self._embedding_size])]  # unclear in the paper to what this matrix is tied

            elif share_type == 'layerwise':

                A = tf.Variable(tf.concat(0, [ nil_word_slot, self._init([self._vocab_size - 1, self._embedding_size]) ]))
                C = tf.Variable(tf.concat(0, [ nil_word_slot, self._init([self._vocab_size - 1, self._embedding_size]) ]))

                self.A = [A * self._hops]
                self.C = [C * self._hops]

                TA = tf.Variable(self._init([self._memory_size, self._embedding_size]), name='TA')
                TC = tf.Variable(self._init([self._memory_size, self._embedding_size]), name='TC')

                self.TA = [TA * self._hops]
                self.TC = [TC * self._hops]

            else:
                raise NotImplementedError

            assert len(self.A) == self._hops
            assert len(self.TA) == self._hops
            assert len(self.C) == self._hops
            assert len(self.TC) == self._hops

        self._nil_vars = set([a.name for a in self.A] + [c.name for c in self.C])

    def _inference(self, stories, observers, queries, share_type):
        """
        Args:
            stories: Tensor (None, memory_size, sentence_size)
            observers: Tensor (None, memory_size, num_caches)
            queries: Tensor (None, sentence_size)

        Returns:
            TODO
        """
        with tf.variable_scope(self._name):

            q_emb = tf.nn.embedding_lookup(self.B, queries)  # B, query embedding matrix
            u_0 = tf.reduce_sum(q_emb * self._encoding, 1)  # embedded query, shape (None, embedding_size)

            u = [u_0]
            self.probs = []
            self.r = []

            observer_stories = tf.einsum('ijk,ijl->ijkl', stories, observers)  # observer-masked stories, shape (None, embedding_size, sentence_size, num_caches)

            for i in range(self._hops):

                m_emb = tf.nn.embedding_lookup(self.A[i], observer_stories)  # A, memory embedding matrix
                m = tf.reduce_sum(tf.einsum('ijklm,km->ijklm', m_emb, self._encoding), 2) + tf.expand_dims(tf.expand_dims(self.TA[i], 0), 2)  # embedded memory, shape (None, memory_size, num_caches, embedding_size)

                c_emb = tf.nn.embedding_lookup(self.C[i], observer_stories)  # C, output embedding matrix
                c = tf.reduce_sum(tf.einsum('ijklm,km->ijklm', c_emb, self._encoding), 2) + tf.expand_dims(tf.expand_dims(self.TC[i], 0), 2)  # embedded output, shape (None, memory_size, num_caches, embedding_size)

                dotted = tf.reduce_sum(tf.einsum('il,ijkl->ijkl', u[-1], m), 3)  # u^T m_i = \sum
                probs = tf.nn.softmax(dotted, dim=1)  # p_i = softmax(u^T m_i), shape (None, memory_size, num_caches)
                self.probs.append(probs)

                u_k = tf.matmul(u[-1], self.H)  # u_{k+1} = (u_k)^T H + ...

                # Sum over memory size
                o_k = tf.reduce_sum(tf.einsum('ijk,ijkl->ijkl', probs, c), [1])  

                # Attend over memory caches
                dotted = tf.reduce_sum(tf.einsum('ik,ijk->ijk', u[-1], o_k), 2)
                r =  tf.nn.softmax(dotted)
                self.r.append(r)

                o_k = tf.reduce_sum(tf.einsum('ij,ijk->ijk', r, o_k), 1)

                u_k = u_k + o_k
                
                # ... + \sum_j o_jk, shape (None, embedding_size)
                #u_k += tf.einsum(tf.reduce_sum(tf.einsum('ijk,ijkl->ijkl', probs, c), [1, 2])  # ... + \sum_j o_jk, shape (None, embedding_size)

                # Nonlinearity
                if self._nonlin:
                    u_k = self._nonlin(u_k)

                u.append(u_k)

            self.probs = tf.pack(self.probs, 3)

            return tf.matmul(u_k, self.W)

    def batch_fit(self, stories, observers, queries, answers):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        feed_dict = {
            self._stories: stories, 
            self._observers: observers, 
            self._queries: queries, 
            self._answers: answers
        }
        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
        return loss

    def predict(self, stories, observers, queries, answers):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {
            self._stories: stories, 
            self._observers: observers, 
            self._queries: queries,
        }
        fetches = [
            self.predict_op,
            self.probs,
            self.r,
            self._stories,
            self._queries,
        ]

        predictions, probs, r, stories, queries = self._sess.run(fetches, feed_dict=feed_dict)

        return predictions, (probs, r)

    def predict_proba(self, stories, queries):
        """Predicts probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries}
        return self._sess.run(self.predict_proba_op, feed_dict=feed_dict)

    def predict_log_proba(self, stories, queries):
        """Predicts log probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries}
        return self._sess.run(self.predict_log_proba_op, feed_dict=feed_dict)

    def reverse_vocab_dict(self, sentence):
        """Returns a sequence of word tokens from a list of integers.
 
        Args:
            sentence: List (sentence_size)
        """
        return [self._reverse_vocab_dict[i] for i in sentence]

    def tex_output(self, predictions, attendance, probs, stories, queries, answers):

        colors = [
            'red', 
            'blue', 
            'green',
            'black',
            'orange',
            'purple',
            'pink',
            'lime',
            'cyan',
            'brown',
            'darkgray',
            'gray',
            'lightgray',
            'magenta',
            'olive',
            'teal',
            'white',
            'violet',
            'yellow',
        ]

        s = ""

        for i in range(predictions.shape[0]):

            if i > 10:
                break

            s += r"""
\begin{figure}[h]
\begin{tikzpicture}
    \begin{axis}[
        xmin=-10, xmax=%d,
        hide axis,
        smooth,
    ]
""" % (self._hops + 1)

            x_bias = 1.

            for n in range(self._hops):

                #coords_to_normalize.append(probs[n][i, j])

                s += r"""
\addplot[color=%s,mark=x] coordinates {""" % colors[n]

                coordinates = []
                texts = []

                #coor = r"""(%f,%d)""" % (x_bias, stories.shape[1] + 1)
                #coordinates.append(coor)

                coords_to_normalize = []
                axis_coord = []

                for j, negative_j in zip(range(stories.shape[1]), reversed(range(stories.shape[1]))):

                    reverse = self.reverse_mapping([stories[i, j, k] for k in range(stories.shape[2])])
                    sent = ' '.join([x for x in reverse if x != 'NIL'])

                    #import pdb; pdb.set_trace()
                    #if i*stories.shape[1] + j == np.argmax(attendance, axis=1):
                        #sent += '**'

                    if not sent:
                        continue

                    #if len(sent) > 30:
                        #sent = sent[:30] + '...'

                    texts.append(r"""\node at (axis cs:-10,%d) [anchor=west] {%s};""" % (negative_j, sent.replace('_', ' ')))

                    coords_to_normalize.append(probs[i, j, n])
                    axis_coord.append(negative_j)

                coords = [x / np.sum(coords_to_normalize + [0.25]) for x in coords_to_normalize]  # add some smoothing so plot does not overlap
                coordinates += [r"""(%.5f,%d)""" % (x_bias + x, y) for x, y in zip(coords, axis_coord)]

                #coor = r"""(%f,%d)""" % (x_bias, stories.shape[1] - len(coordinates) - 1)
                #coordinates.append(coor)
                    
                s += '\n'.join(coordinates)
                s += r"""
};"""


                x_bias += 1

            s += '\n'.join(texts)

            query_sent = ' '.join([x for x in self.reverse_mapping(queries[i]) if x != 'NIL']).replace('_', ' ')
            prediction_word = self.reverse_mapping([predictions[i]])[0].replace('_', ' ')
            answer_word = self.reverse_mapping([answers[i]])[0].replace('_', ' ')

            s += r"""
\end{axis}
\end{tikzpicture}
\caption{\textbf{Query:} %s    \textbf{Prediction:} %s    \textbf{Correct answer:} %s}
\end{figure}
""" % (query_sent, prediction_word, answer_word)

            #if prediction_word != answer_word

        return s

