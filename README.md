# NNVI

## Introduction
A Python implementation of the obstacle problem solver described in the paper **A neural network approach to learning solutions of a class of elliptic variational inequalities**, A. Alphonse, M. Hintermüller, A. Kister, C. H. Lun and C. Sirotenko, [arXiv](https://arxiv.org/pdf/2411.18565).

## Usage
The code is tested with Python 3.11. First install the dependencies by running:

```
pip install -r requirements.txt
```

To start training, run the following command replacing the path to the config file as appropriate:

```
python -m nnvi.train --config configs/1D.yaml
```

A number of different examples have been implemented in `examples.py` which have been named in the `EXAMPLES_MAP` dictionary. One can select which example to run by changing the `example` field in the config files to a named example.

Hyperparameter tuning with the Optuna package is implemented in the file `hyperparameter_tuning_optuna.py`.

## Parallelisation 
To use Ray for parallelising experiments, first install Ray by

```
pip install ray["train"]==2.37.0
```

To run examples on Ray, use

```
python -m train_ray --config configs/1D.yaml
```
