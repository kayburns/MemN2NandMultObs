# Memory Network and Multiple Observer Models

Memory network model originally developed by [Facebook](https://github.com/facebook/MemNN).
Code base was adapted from [domluna's implementation](https://github.com/domluna/memn2n) with major
contributions from [eringrant](https://people.eecs.berkeley.edu/~eringrant/).

This is part of a series of repositories for our 2018 EMNLP Paper.
Please see the [original repository](https://github.com/kayburns/tom-qa-dataset.git) for information on the dataset.

## Getting Started

Install all of the requirements.
- Tensorflow v1.0 or greater.
- Pandas
- Matplotlib
- Numpy
- Parallel: `(wget -O - pi.dk/3 || curl pi.dk/3/ || fetch -o - http://pi.dk/3) | bash`

## Running Experiments
```
./run_tasks.sh
```
This will create a folder inside of `results/` with all of the results, which are processed with the step below.

## Running Analysis
```
python tom_experiments.py --result_files results/${RESULT_FNAME}
```
