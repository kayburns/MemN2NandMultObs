from collections import defaultdict
import glob
#from nltk.util import flatten
import numpy as np
import os
import pandas as pd
from pprint import pprint
import re


results_path = [
    'results/7caccbe/'
]
#babi_data_path = 'data/tasks_1-20_v1-2/'
aug_data_path = 'data/sally_anne/'


## Compute frequency baseline dictionary
#
#def is_number(s):
#    try:
#        float(s)
#        return True
#    except ValueError:
#        pass
# 
#    try:
#        import unicodedata
#        unicodedata.numeric(s)
#        return True
#    except (TypeError, ValueError):
#        pass
#    return False
#
#def count_frequency(words):
#    words = flatten(words)
#    d = defaultdict(int)
#    for word in words:
#        d[word] += 1
#    return d
#
#def extract_answers(text):
#    answers = []
#    for line in text:
#        if '?' in line:  # question
#            a = line[line.index('?') + 1:]
#            answers.append(' '.join(a))
#    return answers
#
#def order_by_frequency(d):
#    return sorted(d.keys(), key=lambda x: d[x], reverse=True)
#
#def read_file(file):
#    with open(file) as f:
#        lines = [x.rstrip('\n').replace(".", "") for x in f.readlines()]
#        lines = [[x.strip() for x in re.split('(\W+)?', sent) if x.strip() and not is_number(x)] for sent in lines]
#    return lines

containers = {}

containers['tiny'] = [
'green_drawer',
'red_crate',
'red_cupboard',
'green_pantry',
]

containers['small'] = [
'green_envelope',
'red_treasure_chest',
'blue_pantry',
'red_crate',
'red_bathtub',
'blue_bottle',
'red_pantry',
'blue_treasure_chest',
'green_bottle',
'green_cupboard',
]

containers['large'] = [
'red_bottle',
'blue_container',
'green_box',
'blue_pantry',
'green_envelope',
'red_bucket',
'red_drawer',
'red_pantry',
'green_basket',
'blue_envelope',
'red_box',
'red_treasure_chest',
'blue_cupboard',
'green_cupboard',
'green_container',
'green_bathtub',
'green_drawer',
'blue_bucket',
'red_cupboard',
'blue_basket',
'red_crate',
'blue_bottle',
'red_bathtub',
'blue_suitcase',
'blue_bathtub',
'blue_box',
'red_suitcase',
'red_bottle',
'blue_container',
'green_box',
'blue_pantry',
'green_envelope',
'red_bucket',
'red_drawer',
'red_pantry',
'green_basket',
'blue_envelope',
'red_box',
'red_treasure_chest',
'blue_cupboard',
'green_cupboard',
'green_container',
'green_bathtub',
'green_drawer',
'blue_bucket',
'red_cupboard',
'blue_basket',
'red_crate',
'blue_bottle',
'red_bathtub',
'blue_suitcase',
'blue_bathtub',
'blue_box',
'red_suitcase',
]

#for folder in glob.glob(os.path.join(aug_data_path, '*')):
#    for train_task_file in glob.glob(os.path.join(folder, '*train.txt')):
#        world_containers = containers[train_task_file.split('_')[2]]
#        test_task_file = train_task_file.rstrip('train.txt') + 'test.txt'
#    
#        train_task = read_file(train_task_file)
#        test_task = read_file(test_task_file)
#
#        # Do not include questions in the frequency count
#        train_freq = count_frequency([line for line in train_task if '?' not in line])
#        test_freq = count_frequency([line for line in test_task if '?' not in line])
#        
#        answers = extract_answers(test_task)       
#    
#        most_freq_train_word = order_by_frequency(train_freq)[0]
#        most_freq_test_word = order_by_frequency(test_freq)[0]
#
#        train_freq_baseline = np.mean([1 if x == most_freq_train_word else 0 for x in answers])
#        test_freq_baseline = np.mean([1 if x == most_freq_test_word else 0 for x in answers])
#
#        # Cumulative memorization frequency baseline
#        responses = []
#        container_responses = []
#        temp = []
#        for line in test_task:
#            if '?' in line:
#                freq_order = order_by_frequency(count_frequency(temp))
#                container_freq_order = [x for x in freq_order if x in world_containers]
#                responses.append(freq_order[0])
#                container_responses.append(container_freq_order[0])
#            else:
#                temp.append(line)
#                
#        assert len(responses) == len(answers)
#        cumulative_test_baseline = np.mean([1 if responses[i] == answers[i] else 0 for i in range(len(answers))])
#        cumulative_container_test_baseline = np.mean([1 if container_responses[i] == answers[i] else 0 for i in range(len(answers))])
    
    
all_results = []
for file in [item for sublist in results_path for item in glob.glob(os.path.join(sublist, '*'))]:
    results = np.load(file).item()
    world_size = results['data_path'].split('/')[-1].split('_')[1]

    if results['num_caches'] == 1:
        continue

    # Baselines:
    if world_size == 'large':
        random_acc = 1. / 150
        world_size = 3
    elif world_size == 'small':
        world_size = 2
        random_acc = 1. / 70
    elif world_size == 'tiny':
        world_size = 1
        random_acc = 1. / 50
    else:
        raise NotImplementedError
        
    # TODO: hack
    true_belief_first_acc_label = [k for k in results.keys() if 'true' in k and 'acc' in k and 'attendance' not in k and 'first' in k]
    false_belief_first_acc_label = [k for k in results.keys() if 'false' in k and 'acc' in k and 'attendance' not in k and 'false_false' not in k and 'first' in k]
    false_false_belief_first_acc_label = [k for k in results.keys() if 'false_false' in k and 'acc' in k and 'attendance' not in k and 'first' in k]
    assert len(true_belief_first_acc_label) == 1
    assert len(false_belief_first_acc_label) == 1
    assert len(false_false_belief_first_acc_label) == 1
    true_belief_first_acc_label = true_belief_first_acc_label[0]
    false_belief_first_acc_label = false_belief_first_acc_label[0]
    false_false_belief_first_acc_label = false_false_belief_first_acc_label[0]

    true_belief_second_acc_label = [k for k in results.keys() if 'true' in k and 'acc' in k and 'attendance' not in k and 'second' in k]
    false_belief_second_acc_label = [k for k in results.keys() if 'false' in k and 'acc' in k and 'attendance' not in k and 'false_false' not in k and 'second' in k]
    false_false_belief_second_acc_label = [k for k in results.keys() if 'false_false' in k and 'acc' in k and 'attendance' not in k and 'second' in k]
    assert len(true_belief_second_acc_label) == 1
    assert len(false_belief_second_acc_label) == 1
    assert len(false_false_belief_second_acc_label) == 1
    true_belief_second_acc_label = true_belief_second_acc_label[0]
    false_belief_second_acc_label = false_belief_second_acc_label[0]
    false_false_belief_second_acc_label = false_false_belief_second_acc_label[0]
    
    true_belief_reality_acc_label = [k for k in results.keys() if 'true' in k and 'acc' in k and 'attendance' not in k and 'reality' in k]
    false_belief_reality_acc_label = [k for k in results.keys() if 'false' in k and 'acc' in k and 'attendance' not in k and 'false_false' not in k and 'reality' in k]
    false_false_belief_reality_acc_label = [k for k in results.keys() if 'false_false' in k and 'acc' in k and 'attendance' not in k and 'reality' in k]
    assert len(true_belief_reality_acc_label) == 1
    assert len(false_belief_reality_acc_label) == 1
    assert len(false_false_belief_reality_acc_label) == 1
    true_belief_reality_acc_label = true_belief_reality_acc_label[0]
    false_belief_reality_acc_label = false_belief_reality_acc_label[0]
    false_false_belief_reality_acc_label = false_false_belief_reality_acc_label[0]
    
    true_belief_memory_acc_label = [k for k in results.keys() if 'true' in k and 'acc' in k and 'attendance' not in k and 'memory' in k]
    false_belief_memory_acc_label = [k for k in results.keys() if 'false' in k and 'acc' in k and 'attendance' not in k and 'false_false' not in k and 'memory' in k]
    false_false_belief_memory_acc_label = [k for k in results.keys() if 'false_false' in k and 'acc' in k and 'attendance' not in k and 'memory' in k]
    assert len(true_belief_memory_acc_label) == 1
    assert len(false_belief_memory_acc_label) == 1
    assert len(false_false_belief_memory_acc_label) == 1
    true_belief_memory_acc_label = true_belief_memory_acc_label[0]
    false_belief_memory_acc_label = false_belief_memory_acc_label[0]
    false_false_belief_memory_acc_label = false_false_belief_memory_acc_label[0]
    
    task_id = max(results['task_ids'])
    assert task_id in [21, 22, 23, 24, 25]

    """
    if results['num_caches'] > 1:

        # Memory cache attendance
        tb_cache_0, tb_cache_1, tb_cache_2 = list(results['true_belief_task_test_test_r'][0].mean(axis=0))
        fb_cache_0, fb_cache_1, fb_cache_2 = list(results['false_belief_task_test_test_r'][0].mean(axis=0))
        ffb_cache_0, ffb_cache_1, ffb_cache_2 = list(results['false_false_belief_task_test_test_r'][0].mean(axis=0))
        tbtom_cache_0, tbtom_cache_1, tbtom_cache_2 = list(results['true_belief_task_tom_test_test_r'][0].mean(axis=0))
        fbtom_cache_0, fbtom_cache_1, fbtom_cache_2 = list(results['false_belief_task_tom_test_test_r'][0].mean(axis=0))
        ffbtom_cache_0, ffbtom_cache_1, ffbtom_cache_2 = list(results['false_false_belief_task_tom_test_test_r'][0].mean(axis=0))
    """

    num_ex = results['data_path'].split('/')[-1].split('_')[3]
    exit_p = results['data_path'].split('/')[-1].split('_')[5]
    search_p = results['data_path'].split('/')[-1].split('_')[7]
    inform_p = results['data_path'].split('/')[-1].split('_')[9]
    all_results.append([
        task_id,
        results[true_belief_first_acc_label],
        results[false_belief_first_acc_label],
        results[false_false_belief_first_acc_label],
        results[true_belief_second_acc_label],
        results[false_belief_second_acc_label],
        results[false_false_belief_second_acc_label],
        results[true_belief_reality_acc_label],
        results[false_belief_reality_acc_label],
        results[false_false_belief_reality_acc_label],
        results[true_belief_memory_acc_label],
        results[false_belief_memory_acc_label],
        results[false_false_belief_memory_acc_label],
        results['dim_memory'],
        results['dim_emb'],
        results['learning_rate'],
        results['num_hops'],
        world_size,
        exit_p,
        search_p,
        inform_p,
        num_ex,
    ])

    """
    if results['num_caches'] > 1:
        all_results.append([
            tb_cache_0,
            tb_cache_1,
            tb_cache_2,
            fb_cache_0,
            fb_cache_1,
            fb_cache_2,
            ffb_cache_0,
            ffb_cache_1,
            ffb_cache_2,
            tbtom_cache_0,
            tbtom_cache_1,
            tbtom_cache_2,
            fbtom_cache_0,
            fbtom_cache_1,
            fbtom_cache_2,
            ffbtom_cache_0,
            ffbtom_cache_1,
            ffbtom_cache_2,
            #random_acc,    
        ])        
    """

all_results = np.stack(all_results)

all_columns = [
    'task ID',
    'true belief first test accuracy',
    'false belief first test accuracy',
    'false false belief first test accuracy',
    'true belief second test accuracy',
    'false belief second test accuracy',
    'false false belief second test accuracy',
    'true belief reality test accuracy',
    'false belief reality test accuracy',
    'false false belief reality test accuracy',
    'true belief memory test accuracy',
    'false belief memory test accuracy',
    'false false belief memory test accuracy',
    'memory size',
    'embedding size',
    'learning_rate',
    'number of hops',
    'world size',
    'exit probability',
    'search probability',
    'inform probability',
    'number of examples',
]

if False:
    all_columns += [
        'true belief test memory cache att cache 0',
        'true belief test memory cache att cache 1',
        'true belief test memory cache att cache 2',
        'false belief test memory cache att cache 0',
        'false belief test memory cache att cache 1',
        'false belief test memory cache att cache 2',
        'false false belief test memory cache att cache 0',
        'false false belief test memory cache att cache 1',
        'false false belief test memory cache att cache 2',
        'true belief theory of mind test memory cache att cache 0',
        'true belief theory of mind test memory cache att cache 1',
        'true belief theory of mind test memory cache att cache 2',
        'false belief theory of mind test memory cache att cache 0',
        'false belief theory of mind test memory cache att cache 1',
        'false belief theory of mind test memory cache att cache 2',
        'false false belief theory of mind test memory cache att cache 0',
        'false false belief theory of mind test memory cache att cache 1',
        'false false belief theory of mind test memory cache att cache 2',
        #'random baseline accuracy',
    ]

df = pd.DataFrame(all_results, columns=all_columns, dtype=float)

world_sizes = [1, 2, 3]
search_probs = [1.0]
#exit_probs = [0.0, 0.5, 1.0]
exit_probs = [0.5]
inform_probs = [0.0]
#tom_probs = [0.5]  # TODO: incorporate testing condition
num_examples = [1000]#, 10000]

task_ids = [21, 22, 23, 24, 25]
dim_memory= [5, 10, 20, 50]
dim_embedding = [5, 10, 20, 50, 100]
num_hops = [1, 2, 3, 4, 5]

tasks_labels = [
    ('true belief first test', 'true_belief_first'),
    ('false belief first test', 'false_belief_first'),
    ('false false belief first test', 'false_false_belief_first'),
    ('true belief second test', 'true_belief_second'),
    ('false belief second test', 'false_belief_second'),
    ('false false belief second test', 'false_false_belief_second'),
    ('true belief reality test', 'true_belief_reality'),
    ('false belief reality test', 'false_belief_reality'),
    ('false false belief reality test', 'false_false_belief_reality'),
    ('true belief memory test', 'true_belief_memory'),
    ('false belief memory test', 'false_belief_memory'),
    ('false false belief memory test', 'false_false_belief_memory'),
]

task_id_labels = {
    21: 'ab',
    22: 'ba',
    23: 'aba',
    24: 'ab_ba',
    25: 'ab_ba_aba',
}


import errno    
import os


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

mkdir_p('analysis')
mkdir_p(os.path.join('analysis', 'pooled'))
mkdir_p(os.path.join('analysis', 'by_embsize'))
mkdir_p(os.path.join('analysis', 'by_memsize'))
mkdir_p(os.path.join('analysis', 'by_hop'))
mkdir_p(os.path.join('analysis', 'att_over_mem'))

for exit_prob in exit_probs:

    for inform_prob in inform_probs:

        #for tom_prob in tom_probs:
        if True:

            inner_df = df[(df['exit probability'] == exit_prob)]
            inner_df = inner_df[(inner_df['inform probability'] == inform_prob)]

            for test_task, test_label in tasks_labels:

                for task_id in task_ids:
                    filename = 'train_%s_exit_%.2f_inform_%.2f_test_%s.csv' % (task_id_labels[task_id], exit_prob, inform_prob, test_label)
                    inner_inner_df = inner_df[(inner_df['task ID'] == task_id)]

                    values = inner_inner_df[test_task + ' accuracy']
                    np.savetxt(os.path.join('analysis', 'pooled', filename), values)

                    """

                    values_0 = inner_inner_df['%s memory cache att cache 0' % test_task]
                    values_1 = inner_inner_df['%s memory cache att cache 1' % test_task]
                    values_2 = inner_inner_df['%s memory cache att cache 2' % test_task]

                    filename = 'train_%s_exit_%.2f_inform_%.2f_tom_%.2f_test_%s_cache_0.csv' % (task_id_labels[task_id], exit_prob, inform_prob, tom_prob, test_label)
                    np.savetxt(os.path.join('analysis', 'att_over_mem', filename), values_0)
                    filename = 'train_%s_exit_%.2f_inform_%.2f_tom_%.2f_test_%s_cache_1.csv' % (task_id_labels[task_id], exit_prob, inform_prob, tom_prob, test_label)
                    np.savetxt(os.path.join('analysis', 'att_over_mem', filename), values_1)
                    filename = 'train_%s_exit_%.2f_inform_%.2f_tom_%.2f_test_%s_cache_2.csv' % (task_id_labels[task_id], exit_prob, inform_prob, tom_prob, test_label)
                    np.savetxt(os.path.join('analysis', 'att_over_mem', filename), values_2)
                    """

                """
                for n in num_hops:
                    filename = 'train_aba_exit_%.2f_test_%s_nhops_%d.csv' % (exit_prob, test_label, n)
                    inner_inner_df = inner_df[(inner_df['number of hops'] == n)]
                    values = inner_inner_df[test_task + ' accuracy']
                    np.savetxt(os.path.join('analysis', 'by_hop', filename), values)

                for n in dim_memory:
                    filename = 'train_aba_exit_%.2f_test_%s_memsize_%d.csv' % (exit_prob, test_label, n)
                    inner_inner_df = inner_df[(inner_df['memory size'] == n)]
                    values = inner_inner_df[test_task + ' accuracy']
                    np.savetxt(os.path.join('analysis', 'by_memsize', filename), values)

                for n in dim_embedding:
                    filename = 'train_aba_exit_%.2f_test_%s_embsize_%d.csv' % (exit_prob, test_label, n)
                    inner_inner_df = inner_df[(inner_df['embedding size'] == n)]
                    values = inner_inner_df[test_task + ' accuracy']
                    np.savetxt(os.path.join('analysis', 'by_embsize', filename), values)
                    """
                   
