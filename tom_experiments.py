import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import itertools

def parse_args():
    parser = argparse.ArgumentParser(
        description='conduct analysis on memn2n results'
    )
    parser.add_argument(
        '--result_files',
        action='append'
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

QUESTIONS = ['memory', 'reality', 'search', 'belief']
TASKS = ['tb', 'fb', 'sofb']

# ------------------ Convert Result Files to DataFrames ---------------------- #

def create_data_frame(results):
    df = pd.DataFrame(results[1:], columns=results[0], dtype=float)
    return df

def generate_result_table_by_question(result_path):
    """Loads numpy files saved from memn2n and returns DataFrame."""
    # we'll use these lists to construct our dataframe
    # the first few entries will contain column names
    val_results = []
    test_results = []

    # first define keys for values we want to extract from saved files
    params = ['dim_memory', 'dim_emb', 'learning_rate', 'num_hops', \
        'num_caches', 'world size', 'num ex', 'noise']

    task_labels_val = [] 
    task_labels_test = []
    for task_type, question in itertools.product(TASKS, QUESTIONS): 
        task_labels_val.append(
            '%s_%s_val_test_test_acc' % (task_type, question)
        )
        task_labels_test.append(
            '%s_%s_test_test_test_acc' % (task_type, question)
        )
    val_results.extend(task_labels_val + params)
    test_results.extend(task_labels_test + params)

    # results of initializations are stored across multiple files
    # iterate through them to collect results for the DataFrame
    for file in (glob.glob(result_path + "*.npy")):
        results = np.load(file).item()
        world_size = results['data_path'].split('/')[-1].split('_')[1]
        noise = results['data_path'].split('/')[-1].split('_')[4]
        num_ex = results['data_path'].split('/')[-1].split('_')[3]

        val_results.extend([results[label] for label in task_labels_val])
        val_results.extend([results[param] for param in params])
        val_results.extend([world_size, num_ex, noise])
        test_results.extend([results[label] for label in task_labels_test])
        test_results.extend([results[param] for param in params])
        test_results.extend([world_size, num_ex, noise])

    num_entries = len(params) + len(task_labels_val)
    val_results = np.reshape(val_results, (-1, num_entries))
    test_results = np.reshape(test_results, (-1, num_entries))

    return create_data_frame(val_results), create_data_frame(test_results)


# ------------------- Analyse Results and Find Best Models ------------------- #


def average_across_tasks(df, split):
    """
    Add new columns to dataframe with overall performance across tasks and
    questions. Will mutate df!
    """
    TB_columns = ['tb_'+q+split+'test_test_acc' for q in QUESTIONS]
    FB_columns = ['fb_'+q+split+'test_test_acc' for q in QUESTIONS]
    SOFB_columns = ['sofb_'+q+split+'test_test_acc' for q in QUESTIONS]
    all_questions = TB_columns + FB_columns + SOFB_columns
    belief_columns = [t+'belief'+split+'test_test_acc' for t in TASKS]
    memory_columns = [t+'memory'+split+'test_test_acc' for t in TASKS]
    reality_columns = [t+'reality'+split+'test_test_acc' for t in TASKS]
    search_columns = [t+'search'+split+'test_test_acc' for t in TASKS]

    df['TB_overall'] = df[TB_columns].mean(axis=1)
    df['FB_overall'] = df[FB_columns].mean(axis=1)
    df['SOFB_overall'] = df[SOFB_columns].mean(axis=1)
    df['overall'] = df[all_questions].mean(axis=1)
    df['memory_overall'] = df[memory_columns].mean(axis=1)
    df['reality_overall'] = df[reality_columns].mean(axis=1)
    df['search_overall'] = df[search_columns].mean(axis=1)
    df['belief_overall'] = df[belief_columns].mean(axis=1)

def find_best_model(df, split):
    """
    Takes in a DataFrame as computed in generate_result_table_by_question and
    finds the performance of the best model within each task and overall.
    Will mutate df by adding more columns!
    Return dictionary with performance and parameters of best models.
    """
    assert split == 'val' or split == 'test'
    df = average_across_tasks(df, split)
    best_models = {}
    best_models['tb_best_'+split] = df_val.loc[df_val['TB_overall'].idxmax()]
    best_models['fb_best_'+spllit] = df_val.loc[df_val['FB_overall'].idxmax()]
    best_models['sofb_best_'+split] = df_val.loc[df_val['SOFB_overall'].idxmax()] 
    best_models['best_'+split] = df_val.loc[df_val['overall'].idxmax()] 
    best_models['memory_best_'+split] = df_val.loc[df_val['memory_overall'].idxmax()]
    best_models['reality_best_'+split] = df_val.loc[df_val['reality_overall'].idxmax()]
    best_models['search_best_'+split] = df_val.loc[df_val['search_overall'].idxmax()]
    best_models['belief_best_'+split] = df_val.loc[df_val['belief_overall'].idxmax()]

    return best_models

def box_plot_creation_helper(df, title, split, save_fig=False):
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(9*3, 4*2))
    belief_columns = [t+'belief'+split+'test_test_acc' for t in TASKS]
    memory_columns = [t+'memory'+split+'test_test_acc' for t in TASKS]
    reality_columns = [t+'reality'+split+'test_test_acc' for t in TASKS]
    search_columns = [t+'search'+split+'test_test_acc' for t in TASKS]
    all_columns = memory_columns + reality_columns + \
        search_columns + belief_columns

    data = plot_list = [df[c] for c in all_columns]

    bplots = []
    medianprops = dict(linewidth=3, color='navy') 
    for i in range(4):
        bplots.appned(ax[i].boxplot(
            data[i*3:(i+1)*3],
            vert=True,  # vertical box alignment
            patch_artist=True,  # fill with color
            medianprops=medianprops,
            labels=labels
        )

    colors = ['lightgray']*3 + ['lightpink']*3 + \
        ['lightpink', 'lightblue', 'lightpink'] + \
        ['lightpink', 'lightblue', 'lightblue']

    for i in range(4):
        for patch, color in zip(bplots[i]['boxes'], colors[i*3:[(i+1)*3]]):
            patch.set_facecolor(color)
        ax[i].set_ylim((0,1))
        ax[i].tick_params(axis = 'both', which = 'major', labelsize = 18)

    fontsize_title = 24
    ax[0].set_title('Memory', fontsize=fontsize_title)
    ax[1].set_title('Reality', fontsize=fontsize_title)
    ax[2].set_title('First Order Belief', fontsize=fontsize_title)
    ax[3].set_title('Second Order Belief', fontsize=fontsize_title)

    if save_fig:
        plt.savefig('plts/'+title+'.png')
    else:
        plt.show()

def create_box_plots():
    titles = []
    for nc, noise, mem in product([1, 3], [10, 0], [50]):
        titles.append(
            str(nc) + '_cache_'+ str(noise) +'_noise_' + str(mem) + '_memory'
        )

        for i in range(len(titles)):
            create_box_plots_single_test_file(
                df_tests[i], titles[i], save_fig=True
            )

if __name__ == '__main__':
    args = parse_args()