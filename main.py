"""TODO"""

from __future__ import absolute_import
from __future__ import print_function


import argparse
from itertools import chain
import logging
import os
from six.moves import range, reduce
import subprocess
import sys

import datetime
import numpy as np
from sklearn import cross_validation, metrics
import tensorflow as tf
    
from data_utils import load_task, vectorize_data
from memn2n import MemN2N
from utils import mkdir_p, bow_encoding, position_encoding

logging.getLogger("tensorflow").setLevel(logging.ERROR)


def load_data(data_dir, task_ids, memory_size, num_caches, random_seed):

    # Load all train and test data
    train, test = [], []
    for i in task_ids:
        tr, te = load_task(data_dir, i)
        train.append(tr)
        test.append(te)
    data = list(chain.from_iterable(train + test))

    vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a + ['.']) for s, _, q, a, _ in data)))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    reverse_word_idx = ['NIL'] + sorted(word_idx.keys(), key=lambda x: word_idx[x])
    
    max_story_size = max(map(len, (s for s, _, _, _, _ in data)))
    mean_story_size = int(np.mean([ len(s) for s, _, _, _, _ in data ]))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _, _, _ in data)))
    query_size = max(map(len, (q for _, _, q, _, _ in data)))
    memory_size = min(memory_size, max_story_size)
    vocab_size = len(word_idx) + 1  # +1 for the NIL word
    sentence_size = max(query_size, sentence_size)  # for the position
    
    logging.info("Longest sentence length: %d" % sentence_size)
    logging.info("Longest story length: %d" % max_story_size)
    logging.info("Average story length: %d" % mean_story_size)

    # Train/validation/test splits
    trainS = []
    valS = []
    trainO = []
    valO = []
    trainQ = []
    valQ = []
    trainA = []
    valA = []
    trainL = []
    valL = []
    for task in train:
        S, O, Q, A, L = vectorize_data(task, word_idx, sentence_size, memory_size, num_caches)
        ts, vs, to, vo, tq, vq, ta, va, tl, vl = cross_validation.train_test_split(S, O, Q, A, L, test_size=0.1, random_state=random_seed)
        trainS.append(ts)
        trainO.append(to)
        trainQ.append(tq)
        trainA.append(ta)
        trainL.append(tl)
        valS.append(vs)
        valO.append(vo)
        valQ.append(vq)
        valA.append(va)
        valL.append(vl)
    
    trainS = reduce(lambda a, b : np.vstack((a, b)), (x for x in trainS))
    trainO = reduce(lambda a, b : np.vstack((a, b)), (x for x in trainO))
    trainQ = reduce(lambda a, b : np.vstack((a, b)), (x for x in trainQ))
    trainA = reduce(lambda a, b : np.vstack((a, b)), (x for x in trainA))
    trainL = reduce(lambda a, b : np.vstack((a, b)), (x for x in trainL))
    valS = reduce(lambda a, b : np.vstack((a, b)), (x for x in valS))
    valO = reduce(lambda a, b : np.vstack((a, b)), (x for x in valO))
    valQ = reduce(lambda a, b : np.vstack((a, b)), (x for x in valQ))
    valA = reduce(lambda a, b : np.vstack((a, b)), (x for x in valA))
    valL = reduce(lambda a, b : np.vstack((a, b)), (x for x in valL))
    
    testS, testO, testQ, testA, testL = vectorize_data(list(chain.from_iterable(test)), word_idx, sentence_size, memory_size, num_caches)
    
    logging.info("Training set shape: %s" % str(trainS.shape))
    
    train_data = trainS, trainO, trainQ, trainA, trainL
    val_data = valS, valO, valQ, valA, valL
    test_data = testS, testO, testQ, testA, testL

    return train_data, val_data, test_data, word_idx, reverse_word_idx, vocab_size, sentence_size, memory_size

    
def train_loop(model, train_data, val_data, batch_size, num_epochs, val_freq):

    trainS, trainO, trainQ, trainA, trainL = train_data
    valS, valO, valQ, valA, valL = val_data

    train_labels = np.argmax(trainA, axis=1)
    val_labels = np.argmax(valA, axis=1)

    n_train = trainS.shape[0]
    n_val = valS.shape[0]

    logging.info("Training Size: %d" % n_train)
    logging.info("Validation Size: %d" % n_val)
    
    batches = zip(range(0, n_train - batch_size, batch_size), range(batch_size, n_train, batch_size))
    batches = [(start, end) for start, end in batches]

    for t in range(1, num_epochs + 1):

        np.random.shuffle(batches)

        total_cost = 0.0
        for start, end in batches:
            s = trainS[start:end]
            o = trainO[start:end]
            q = trainQ[start:end]
            a = trainA[start:end]
            cost_t = model.batch_fit(s, o, q, a)
            total_cost += cost_t

        if val_freq is not None and t % val_freq == 0:

            train_preds = []
            for start in range(0, n_train, batch_size):
                end = start + batch_size
                s = trainS[start:end]
                o = trainO[start:end]
                q = trainQ[start:end]
                a = train_labels[start:end]
                pred, _ = model.predict(s, o, q, a)
                train_preds += list(pred)
            train_preds = np.array(train_preds)

            val_preds, _ = model.predict(valS, valO, valQ, val_labels)
            train_acc = metrics.accuracy_score(train_preds, train_labels)
            val_acc = metrics.accuracy_score(val_preds, val_labels)

            logging.info('-----------------------')
            logging.info('Epoch: %d' % t)
            logging.info('Total Cost: %.2f' % total_cost)
            logging.info('Training Accuracy: %.2f' % train_acc)
            logging.info('Validation Accuracy: %.2f' % val_acc)


def evaluate(model, test_data, out_path):

    testS, testO, testQ, testA, testL = test_data
    test_labels = np.argmax(testA, axis=1)

    n_test = testS.shape[0]
    logging.info("Testing Size: %d" % n_test)

    test_preds, test_probs = model.predict(testS, testO, testQ, test_labels)

    test_acc = metrics.accuracy_score(test_preds, test_labels)
    logging.info("Testing Accuracy: %.2f" % test_acc)

    # TODO deal with observer case
    test_attendance_accs = []
    for i in range(test_probs.shape[3]):
        test_attendance_inner_accs = []
        for j in range(test_probs.shape[2]):
            test_attendance_acc = metrics.accuracy_score(np.argmax(test_probs[:, :, j, i], axis=1), np.argmax(testL, axis=1))
            test_attendance_inner_accs.append(test_attendance_acc)
            print("Testing Accuracy of Attendance at hop %d for observer %d:" % (i, j), test_attendance_acc)
        test_attendance_accs += [np.stack(test_attendance_inner_accs)]
    test_attendance_accs = np.stack(test_attendance_accs)

    return test_acc, test_attendance_accs, test_preds, test_probs


def parse_args(args):

    parser = argparse.ArgumentParser(description='Process command-line arguments.')

    parser.add_argument('-l', '--logging', type=str, default='INFO', dest='logging',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')

    parser.add_argument('-rs', '--random_seed', dest='random_seed', type=int, 
                        default=1,
                        help='An integer random seed value')

    # MODEL HYPERPARAMETERS
    parser.add_argument('-nh', '--num_hops', dest='num_hops', type=int, 
                        default=3,
                        help='Number of memory hops in the model')

    parser.add_argument('-et', '--encoding_type', type=str, default='position_encoding',
                        choices=['position_encoding', 'bow_encoding'],
                        help='The type of encoding to use')

    parser.add_argument('-st', '--share_type', type=str, default='adjacent',
                        choices=['adjacent', 'layerwise'],
                        help='The type of weight tying')

    parser.add_argument('-nl', '--nonlin', type=str, default=None,
                        choices=['relu'],
                        help='The type of encoding to use')

    parser.add_argument('-te', '--temporal_encoding', dest='temporal_encoding', action='store_true',
                        help='Whether to use the temporal encoding')

    parser.add_argument('-dm', '--dim_memory', dest='dim_memory', type=int, 
                        default=50,
                        help='The dimensionality of the memory')

    parser.add_argument('-de', '--dim_emb', dest='dim_emb', type=int, 
                        default=20,
                        help='The dimensionality of the embedding')

    parser.add_argument('-nc', '--num_caches', dest='num_caches', type=int, default=1,
                        help='The maximum number of external memory caches')

    # TRAINING HYPERPARAMETERS
    parser.add_argument('-is', '--init_stddev', dest='init_stddev', type=float, 
                        default=0.1,
                        help='Initial stddev')

    parser.add_argument('-lr', '--learning_rate', dest='learning_rate', type=float, 
                        default=0.01,
                        help='Initial learning rate')

    parser.add_argument('-gn', '--max_grad_norm', dest='max_grad_norm', type=float, 
                        default=40.,
                        help='Number of examples in each minibatch')

    parser.add_argument('-bs', '--batch_size', dest='batch_size', type=int, 
                        default=32,
                        help='Number of examples in each minibatch')

    parser.add_argument('-vf', '--val_freq', dest='val_freq', type=int, 
                        default=None,
                        help='Validate the model every vf epochs')

    parser.add_argument('-ne', '--num_epochs', dest='num_epochs', type=int, 
                        default=100,
                        help='The maximum number of epochs for which to train')

    # DATA
    parser.add_argument('-t', '--task_id', dest='task_ids', type=int, 
                        required=True, action='append',
                        help='The tasks to be tested')

    parser.add_argument('--joint', dest='joint', action='store_true',
                        help='Whether to train on tasks jointly')

    parser.add_argument('-d', '--data_path', dest='data_path', type=mkdir_p,
                        required=True, 
                        help='Path to the folder containing the train and test data for each task')

    parser.add_argument('-o', '--output_dir_path', dest='output_dir_path', type=mkdir_p,
                        default='results',
                        help='Output directory path')

    parser.add_argument('-ws', '--world_size', dest='world_size', type=int, 
                        choices=['tiny', 'small', 'large'],
                        help='TODO')

    parser.add_argument('-sp', '--search_prob', dest='search_prob', type=float, 
                        help='TODO')

    parser.add_argument('-ep', '--exit_prob', dest='exit_prob', type=float, 
                        help='TODO')

    return parser.parse_args(args)


def get_git_revision_short_hash():
    return str(subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])).rstrip('\n')


def main(args=sys.argv[1:]):

    args = parse_args(args)
    logging.basicConfig(level=args.logging, format='%(asctime)s\t%(levelname)-8s\t%(message)s')

    output_path = os.path.join(
            args.output_dir_path,
            #'%s_%s' % (get_git_revision_short_hash(), datetime.datetime.now().time().isoformat())
            datetime.datetime.now().time().isoformat(),
    )

    tf.set_random_seed(args.random_seed)

    if not args.joint:

        for task_id in args.task_ids:

            train_data, val_data, test_data, word_idx, reverse_word_idx, vocab_size, sentence_size, memory_size = load_data(args.data_path, [task_id], args.dim_memory, args.num_caches, args.random_seed)

            optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

            encodings = {
                'bow_encoding': bow_encoding,
                'position_encoding': position_encoding,
            }

            nonlins = {
                'relu': tf.nn.relu,
            }
            if args.nonlin is not None:
                nonlin = nonlins[args.nonlin]
            else:
                nonlin = None
    
            model = MemN2N(args.batch_size, 
                           vocab_size,
                           sentence_size,
                           memory_size,
                           args.num_caches,
                           args.dim_emb,
                           word_idx,
                           reverse_word_idx,
                           args.num_hops,
                           args.max_grad_norm,
                           nonlin=nonlin,
                           optimizer=optimizer,
                           initializer=tf.random_normal_initializer(stddev=args.init_stddev),
                           encoding=encodings[args.encoding_type],
                           temporal_encoding=args.temporal_encoding,
                          )
        
            train_loop(model, 
                       train_data, val_data,
                       args.batch_size, 
                       args.num_epochs, 
                       args.val_freq
                      )

            test_acc, test_attendance_acc, test_preds, test_probs = evaluate(model, test_data, output_path)

            d = {
                'test_preds': test_preds, 
                'test_probs': test_probs, 
                'test_acc': test_acc, 
                'test_attendance_accs': test_attendance_acc,
                'vocab_dict': word_idx,
            }

            d.update(**vars(args))

            np.save(output_path, d)

            tf.reset_default_graph()

    else:
        raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
