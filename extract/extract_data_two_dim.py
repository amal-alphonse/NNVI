from pathlib import Path
import shutil

import torch
import torch.nn as nn


import nnvi.eval_ray_run as eval_r
import pandas as pd

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

def transform_two_d_data(d):
    if len(d['X_x'].shape)>1:
        transformed = {'X_x':d['X_x'].flatten(),
                        'X_y':d['X_y'].flatten(),  
                        'value':d['value'].flatten(),}
    else:
        transformed = d
    return transformed


#base_folder = Path('/home/akister/VI/NN-VI/overleaf_config_identical_confs/')
#base_folder = Path('/home/akister/VI/NN-VI/overleaf_config_slowed_down/')
#base_folder =  Path('/home/akister/VI/NN-VI/overleaf_config_identical_less_force/')
#base_folder = Path('/home/akister/VI/NN-VI/overleaf_config_identical_no_force/')
#base_folder =Path('/home/akister/VI/NN-VI/overleaf_config_identical_very_big_force/')
#base_folder =Path('/home/akister/VI/NN-VI/overleaf_config_identical_strong_obst/')
#base_folder = Path('/home/akister/VI/NN-VI/overleaf_config_identical_weak_obst/')
base_folder = Path('/home/akister/ray_results/for_paper/fn_2024-09-17_13-48-03')
data_folder = base_folder/#'results'
tikz_csvs = base_folder/'tikz_csvs'
tikz_csvs.mkdir(exist_ok=True)

all_results = eval_r.ray_grid(base_folder)


for name in all_results.get_example_names():
    results = all_results.get_results_for_example(example_name=name)
    exp_name = results.get_example_name()
    pdim = results.get_dim()
    
    experiment = results.get_best_experiment(metric='error_l2')
    adress_of_cfg = experiment.cfg.cfg
    grid = experiment.get_regular_grid(n_interior=pdim2nb_grid_points[pdim],n_boundary=10)
    example = experiment.get_example()
    soln_nn=experiment.get_soln()
    gt=example.exact_solution
    fn_s={'sol_nn':soln_nn,
          'gt_nn':gt,
          'diff': diff(soln_nn,gt),
          'test_nn':experiment.get_test(),
          'obstacle_nn': example.obstacle,}
    fn_s_for_second_derivatives = {
        'sol_nn_2_derivative':soln_nn,
        'gt_nn_2_derivative':gt,}
    
    data=[]
    for name,fn in fn_s.items():
        fn_data=eval_r.plot_data_for_nn_on_mesh(fn,grid,pdim=pdim,name=name)
        data.append(fn_data)
    """for name,fn in fn_s_for_second_derivatives.items():
        device='cpu'
        x=grid.get_interior_points().to(device)
        x.requires_grad_()
        y=fn(x)
        grad_output = torch.ones_like(y)
        first_derivative = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=grad_output, create_graph=True)[0]
        second_derivative = torch.autograd.grad(outputs=first_derivative, inputs=x, grad_outputs=grad_output, create_graph=False)[0]
        data.append({'value':list(second_derivative.detach().to('cpu').numpy().flatten()),
                     'X_x': list(x.detach().to('cpu').numpy().flatten()),
                     'name': list([name]*len(x))})
    """
    save_to_folder=tikz_csvs/f'result_best_run_{exp_name}'
    save_to_folder.mkdir(exist_ok=True)
    shutil.copy(src=adress_of_cfg, dst=save_to_folder/'cfg.json')
    dfs=[{'data':pd.DataFrame(transform_two_d_data(entry)),'name':entry['name'][0]} for entry in data]
    for df in dfs:
        name = df['name']
        df['data'].to_csv(save_to_folder/f'{name}.csv')
    
    metric = ['error_l2','error_linfty','loss_r']
    data=[]
    save_to_folder=tikz_csvs/f'train_trajectories_{exp_name}'
    save_to_folder.mkdir(exist_ok=True)
    for id,experiment_id in enumerate(results.get_experiments().keys()):
        experiment = results.get_experiments()[experiment_id]
        progress_data = experiment.get_training_process()
        df=pd.DataFrame(progress_data[['epoch']+metric])
        df.to_csv(save_to_folder/f'{id}.csv')
