from __future__ import absolute_import

import logging
import os
import re
import numpy as np


def load_task(data_dir, task_id, only_supporting=False):
    '''Load the nth task.

    Returns a tuple containing the training and testing data for the task.
    '''
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'qa{}_'.format(task_id)
    train_file = [f for f in files if s in f and 'train' in f][0]
    test_file = [f for f in files if s in f and 'test' in f][0]
    logging.info("Loading train from %s..." % train_file)
    logging.info("Loading test from %s..." % test_file)
    train_data = get_stories(train_file, only_supporting)
    test_data = get_stories(test_file, only_supporting)
    return train_data, test_data


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []

    for line in lines:

        line = str.lower(line)
        nid, line = line.split(' ', 1)
        nid = int(nid)

        if nid == 1:
            story = []
            substory_counter = 0

        if '?' in line:  # question

            if not story:
                raise Exception

            q, a, supporting = line.split('\t')
            supporting = map(int, supporting.split())

            # Remove question marks
            q = q.replace("?", "")
            q = tokenize(q)

            # Answer is one vocab word even if it's actually multiple words
            a = [a]

            substory = None

            if only_supporting:
                # Only select the related substory
                substory = [story[i - 1] for i in supporting]

            else:
                # Provide all the substories
                substory = [x for x in story if x]

            # Extract the observer information
            substory, observers = zip(*substory)

            supporting = [x - substory_counter for x in supporting]

            data.append((substory, observers, q, a, supporting))
            substory_counter += 1
            story.append('')

        else:  # story line
            
            # Check for observer labels
            try:
                line, observers = line.split('\t')
                observers = map(int, observers.split())
            except ValueError:
                observers = None

            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append((sent, observers))

    return data


def get_stories(f, only_supporting=False):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_stories(f.readlines(), only_supporting=only_supporting)


def vectorize_data(data, word_idx, sentence_size, memory_size, num_caches):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    S = []
    O = []
    Q = []
    A = []
    L = []

    for story, observers, query, answer, support in data:

        # STORY LINES
        ss = []
        for i, sentence in enumerate(story, 1):

            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        # OBSERVER FLAGS
        observer_flag_unused = False
        observer_flag_present = False
        observers = observers[::-1][:memory_size][::-1]
        o = np.zeros((memory_size, num_caches))
        for i, x in enumerate(observers):
            o[i, 0] = 1  # the oracle observer
            if x is not None:
                for j in x:
                    assert j > 0
                    try:
                        assert j < num_caches
                        o[i, j] = 1
                        observer_flag_present = True
                    except AssertionError:
                        observer_flag_unused = True

        if num_caches > 1 and not observer_flag_present:
            logging.warning('Observer flags not present but number of caches > 1.')
            o = np.ones((memory_size, num_caches))
            observer_flag_unused = False

        if observer_flag_unused:
            logging.warning('Observer flags present but unused.')
            o = np.ones((memory_size, num_caches))

        # QUERIES
        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        # ANSWERS
        y = np.zeros(len(word_idx) + 1) # 0 is reserved for nil word
        for a in answer:
            y[word_idx[a]] = 1

        # SUPPORTING SENTENCES
        l = np.zeros(memory_size)
        for supp in support:
            if supp <= memory_size:
                l[supp - 1] = 1

        S.append(ss)
        O.append(o)
        Q.append(q)
        A.append(y)
        L.append(l)

    return np.array(S), np.array(O), np.array(Q), np.array(A), np.array(L)
