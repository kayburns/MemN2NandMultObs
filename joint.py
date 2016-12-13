"""Example running MemN2N on a single bAbI task.
Download tasks from facebook.ai/babi """
from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_task, vectorize_data
from sklearn import cross_validation, metrics
from memn2n import MemN2N
from itertools import chain
from six.moves import range, reduce

import tensorflow as tf
import numpy as np
import pandas as pd

tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 100, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 40, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "data/tasks_1-20_v1-2_w_metareasoning/en/", "Directory containing bAbI tasks")
tf.flags.DEFINE_string("output_file", "scores.csv", "Name of output file for final bAbI accuracy scores.")
FLAGS = tf.flags.FLAGS

print("Started Joint Model")

n_tasks = 44

# load all train/test data
ids = range(1, n_tasks+1)
train, test = [], []
for i in ids:
    tr, te = load_task(FLAGS.data_dir, i)
    train.append(tr)
    test.append(te)
data = list(chain.from_iterable(train + test))

vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

max_story_size = max(map(len, (s for s, _, _ in data)))
mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
query_size = max(map(len, (q for _, q, _ in data)))
memory_size = min(FLAGS.memory_size, max_story_size)
vocab_size = len(word_idx) + 1 # +1 for nil word
sentence_size = max(query_size, sentence_size) # for the position

print("Longest sentence length", sentence_size)
print("Longest story length", max_story_size)
print("Average story length", mean_story_size)

# train/validation/test sets
trainS = []
valS = []
trainQ = []
valQ = []
trainA = []
valA = []
for task in train:
    S, Q, A = vectorize_data(task, word_idx, sentence_size, memory_size)
    ts, vs, tq, vq, ta, va = cross_validation.train_test_split(S, Q, A, test_size=0.1, random_state=FLAGS.random_state)
    trainS.append(ts)
    trainQ.append(tq)
    trainA.append(ta)
    valS.append(vs)
    valQ.append(vq)
    valA.append(va)

trainS = reduce(lambda a,b : np.vstack((a,b)), (x for x in trainS))
trainQ = reduce(lambda a,b : np.vstack((a,b)), (x for x in trainQ))
trainA = reduce(lambda a,b : np.vstack((a,b)), (x for x in trainA))
valS = reduce(lambda a,b : np.vstack((a,b)), (x for x in valS))
valQ = reduce(lambda a,b : np.vstack((a,b)), (x for x in valQ))
valA = reduce(lambda a,b : np.vstack((a,b)), (x for x in valA))

testS, testQ, testA = vectorize_data(list(chain.from_iterable(test)), word_idx, sentence_size, memory_size)

n_train = trainS.shape[0]
n_val = valS.shape[0]
n_test = testS.shape[0]

print("Training Size", n_train)
print("Validation Size", n_val)
print("Testing Size", n_test)

print(trainS.shape, valS.shape, testS.shape)
print(trainQ.shape, valQ.shape, testQ.shape)
print(trainA.shape, valA.shape, testA.shape)

train_labels = np.argmax(trainA, axis=1)
test_labels = np.argmax(testA, axis=1)
val_labels = np.argmax(valA, axis=1)

tf.set_random_seed(FLAGS.random_state)
batch_size = FLAGS.batch_size
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, epsilon=FLAGS.epsilon)

# This avoids feeding 1 task after another, instead each batch has a random sampling of tasks
batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
batches = [(start, end) for start,end in batches]

with tf.Session() as sess:
    model = MemN2N(batch_size, vocab_size, sentence_size, memory_size, FLAGS.embedding_size, session=sess,
                   hops=FLAGS.hops, max_grad_norm=FLAGS.max_grad_norm, optimizer=optimizer)
    for i in range(1, FLAGS.epochs+1):
        np.random.shuffle(batches)
        total_cost = 0.0
        for start, end in batches:
            s = trainS[start:end]
            q = trainQ[start:end]
            a = trainA[start:end]
            cost_t = model.batch_fit(s, q, a)
            total_cost += cost_t

        if i % FLAGS.evaluation_interval == 0:
            train_accs = []
            step = int(n_train/n_tasks)
            for start in range(0, n_train, step):
                end = start + step
                s = trainS[start:end]
                q = trainQ[start:end]
                pred, human_readable = model.predict(s, q)
                acc = metrics.accuracy_score(pred, train_labels[start:end])
                train_accs.append(acc)

            val_accs = []
            step = int(n_val/n_tasks)
            for start in range(0, n_val, step):
                end = start + step
                s = valS[start:end]
                q = valQ[start:end]
                pred, human_readable = model.predict(s, q)
                acc = metrics.accuracy_score(pred, val_labels[start:end])
                val_accs.append(acc)

            test_accs = []
            step = int(n_test/n_tasks)
            for start in range(0, n_test, step):
                end = start + step
                s = testS[start:end]
                q = testQ[start:end]
                pred, human_readable = model.predict(s, q)
                acc = metrics.accuracy_score(pred, test_labels[start:end])
                test_accs.append(acc)

            print('-----------------------')
            print('Epoch', i)
            print('Total Cost:', total_cost)
            print()
            t = 1
            for t1, t2, t3 in zip(train_accs, val_accs, test_accs):
                print("Task {}".format(t))
                print("Training Accuracy = {}".format(t1))
                print("Validation Accuracy = {}".format(t2))
                print("Testing Accuracy = {}".format(t3))
                print()
                t += 1
            print('-----------------------')

        # Write final results to csv file
        if i == FLAGS.epochs:
            print('Writing final results to {}'.format(FLAGS.output_file))
            df = pd.DataFrame({
            'Training Accuracy': train_accs,
            'Validation Accuracy': val_accs,
            'Testing Accuracy': test_accs
            }, index=range(1, n_tasks+1))
            df.index.name = 'Task'
            df.to_csv(FLAGS.output_file)
