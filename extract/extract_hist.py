from pathlib import Path

import torch
import torch.nn as nn


import nnvi.eval_ray_run as eval_r
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import subprocess
import os
import re

pdim2nb_grid_points = {1: 1000, 2:1000}
def diff(sol,gt):
    if sol is not None and gt is not None:
        def fn(x):
            return sol(x)-gt(x)
    else:
        def fn(x):
            return None
    return fn

#base_folder = Path('/home/akister/VI/NN-VI/overleaf_config_identical_confs/')
#base_folder = Path('/home/akister/VI/NN-VI/overleaf_config_slowed_down/')
#base_folder =  Path('/home/akister/VI/NN-VI/overleaf_config_identical_less_force/')
#base_folder = Path('/home/akister/VI/NN-VI/overleaf_config_identical_no_force/')
#base_folder =Path('/home/akister/VI/NN-VI/overleaf_config_identical_very_big_force/')
#base_folder =Path('/home/akister/VI/NN-VI/overleaf_config_identical_strong_obst/')
#base_folder = Path('/home/akister/VI/NN-VI/overleaf_config_identical_weak_obst/')
#base_folder = Path('/home/akister/NN-VI/one_dim_exps/')
base_folder = Path('/home/akister/NN-VI/one_dim_exps_for_hist/')
data_folder = base_folder/'results'
tikz_csvs = base_folder/'tikz_csvs'
tikz_csvs.mkdir(exist_ok=True)

run_s = [fn.name for fn in data_folder.iterdir() if fn.name.startswith('fn_')]

max_len = 100

for name in run_s:
    results = eval_r.ray_results(data_folder,
                                 name,
                                 max_len=max_len,
                                 use_existing_cash=False)
    exp_name = results.get_example_name()
    pdim = results.get_dim()
    
    experiment = results.get_best_experiment(metric='error_l2')
    grid = experiment.get_regular_grid(n_interior=pdim2nb_grid_points[pdim],n_boundary=10)
    example = experiment.get_example()
    soln_nn=experiment.get_soln()
    gt=example.exact_solution
    
    metric = ['error_l2','error_linfty','loss_r', 'loss1','loss2']
    data=[]
    data_kde=[]
    save_to_folder=tikz_csvs/f'hist_{exp_name}'
    save_to_folder.mkdir(exist_ok=True)
    for id,experiment_id in enumerate(results.get_experiments().keys()):
        experiment = results.get_experiments()[experiment_id]
        errors={}
        errors['id']=id
        errors['experiment_id']=experiment_id

        
        for me in metric:
            final_err = experiment.get_final_error(me)
            errors[me]=final_err          
        data.append(errors)
    df=pd.DataFrame(data)
    kde_of_err={}
        
    for me in metric:
        final_err = df[me]
        kde = gaussian_kde(final_err)
        x_grid = np.linspace(min(final_err), max(final_err), 1000)
        density = kde.evaluate(x_grid)
        kde_of_err[me]=density
        kde_of_err[me+'_x_grid']=x_grid
    df_kde=pd.DataFrame(kde_of_err)
    df.to_csv(save_to_folder/'hist.csv')
    df_kde.to_csv(save_to_folder/'kde.csv')
