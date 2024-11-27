import os
import random
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import copy 

from collections import deque

import ray
from ray import train, tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search import Repeater
from dataclasses import replace
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import MedianStoppingRule

import time
from typing import Dict, Optional, Any



import nnvi as root_module
from torch import Tensor, optim


from nnvi.grids.domain import Domain
from nnvi.config import Config
from nnvi.examples import EXAMPLES_MAP, Example
from nnvi.loss import BaseLoss, build_loss

#from nnvi.grids.grid import * 
from nnvi.grids.grid import Grid, ImportanceForSolution, ImportanceForTest

from nnvi.metrics import L2Error, LInftyError, Metric, Visualise, H_one_Norm
from nnvi.models import build_model

from nnvi.callbacks import Callback, SolutionGradient, TensorboardLogger, Tunes, CheckpointLogger
from nnvi.grids.grid import SimpleGrid
from nnvi.train import train_cfg


from pathlib import Path
import json
from dataclasses import dataclass,asdict

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

 
############ RAY ###################
def fusion_config_dict_ray_dict(config_dict,ray_dict):
    """
    Objects of type Config and the dicts returned by ray have a different strucutr (Configs have susbtructures, ray is flat). We assume that config_dict represents all parameters while ray_dict contains only a selection.
    The fused object should have the type Config and the parameters should be updated according to ray_dict.
    """

    # Find the field that should be updated
    fields_for_update = ray_dict.keys()
    name_of_dict_fields = {
        "gamma_sol":["lr_scheduler_soln_args","gamma"],
        "step_size_sol":["lr_scheduler_soln_args","step_size"],
        "gamma_test":["lr_scheduler_testfn_args","gamma"],
        "step_size_test":["lr_scheduler_testfn_args","step_size"], 
        "width_sol":["soln_model_args","width"],
        "depth_sol":["soln_model_args","depth"],
        "hc_bias":["soln_model_args","hc_bias"],
    }
    # Update the parameters
    updated_config = config_dict
    for k,v in ray_dict.items():
        if k in name_of_dict_fields.keys():
            field_key = name_of_dict_fields[k][0]
            field = getattr(updated_config,field_key)
            field[name_of_dict_fields[k][1]]=v
            setattr(updated_config, field_key, field)
        else:
            setattr(updated_config,k,v)
    return updated_config







def single_ray_train_run(cfg):
    def fn(ray_cfg):
        copy_of_cfg=copy.deepcopy(cfg)
        ray_cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        copy_of_cfg.chpt_folder = os.getcwd()
        copy_of_cfg.seed = np.random.randint(low=2, high=99999)
        copy_of_cfg = fusion_config_dict_ray_dict(copy_of_cfg,ray_cfg)
        error_l2=train_cfg(copy_of_cfg)
        return {"error_l2": error_l2}
    return fn

chin_search_space={
    "seed":tune.uniform(1,123445),
}

def train_ray_importance(cfg: Config) -> None:
    ray.init(dashboard_port = 8097)
    base_cfg = copy.deepcopy(cfg)
    seed_everything(int(base_cfg.seed))
    algo = BayesOptSearch(utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0},
                          #points_to_evaluate = points_to_evaluate_for_ns_two_dim,
                          random_search_steps=200,
                          verbose = 1,
                          patience = 200,
                          skip_duplicate = False)
    algo = Repeater(algo,repeat=100)
    search_space=chin_search_space    


    base_cfg.use_ray = True
    run = single_ray_train_run(base_cfg)
    
    
    trainable_with_gpu = tune.with_resources(run, {"cpu":1.0,"gpu": 0.10})

    tuner = tune.Tuner(
        trainable_with_gpu,
        tune_config=tune.TuneConfig(
        metric="error_l2", #"best_l2"
        mode="min",
        search_alg=algo,
        num_samples=10, #4
        ),
        run_config=train.RunConfig(stop={"time_total_s": 1200000000},sync_config=train.SyncConfig(sync_artifacts=True),),
        param_space=search_space,
    )
    
    results = tuner.fit()


    
#grid_search_space={
    #"example":tune.grid_search(["KS","KS_ns","NSV_two_dim","MT","two_dim"])#,]),"ns_two_dim"
    #}

#grid_search_space={
#    "example":tune.grid_search(["MT","two_dim"])#,]),"ns_two_dim"
#}


#grid_search_space={
#    "example":tune.grid_search(["two_dim"])#,]),"ns_two_dim"
#}

grid_search_space={"weight_soln_obs":tune.grid_search([50,500,1500,3000,5000,8000,9000,10000,12500,15000])}
#grid_search_space={
#    "example":tune.grid_search(["ns_one_dim","one_dim","CM"]),
#}

#grid_search_space={
#    "weight_gap_term":tune.grid_search([5.0e-20,5.0e-18,5.0e-15,5.0e-13,5.0e-12,5.0e-11,5.0e-10,5.0e-9,
#                                         1.0e-8,5.0e-8,1.0e-7,5.0e-7,5.0e-6,
#                                         0.00005,5.0e-4,1.0e-3,5.0e-3,0.01,0.05,0.5])#,]),"ns_two_dim"
#}

def train_ray_grid(cfg: Config, ray_temp_dir: Optional[str] = None) -> None:
    ray.init(dashboard_port = 8097,_temp_dir = ray_temp_dir)
    base_cfg = copy.deepcopy(cfg)

    algo = BasicVariantGenerator()
    search_space=grid_search_space    

    base_cfg.use_ray = True
    run = single_ray_train_run(base_cfg)
    
    
    trainable_with_gpu = tune.with_resources(run, {"cpu":1.0,"gpu": 0.10 if torch.cuda.is_available() else 0.0})

    tuner = tune.Tuner(
        trainable_with_gpu,
        tune_config=tune.TuneConfig(
        metric="error_l2", 
        search_alg=algo,
        num_samples=50 , #4
        ),
        run_config=train.RunConfig(stop={"time_total_s": 1200000000},sync_config=train.SyncConfig(sync_artifacts=True),),
        param_space=search_space,
    )
    
    results = tuner.fit()

optuna_search_space={
    "weight_soln_obs":tune.loguniform(10,1000),
    "weight_gap_term":tune.loguniform(0.00001,0.09),
    "lr_soln":tune.loguniform(0.001,0.1)
}

points_to_evaluate = [{'weight_soln_obs': 991.101792528117, 'weight_gap_term': 0.00014543644926401608, 'lr_soln': 0.00815691946887036},
    {'weight_soln_obs': 842.2053114899255, 'weight_gap_term': 5.249337110482551e-05, 'lr_soln': 0.0032635484853718267},# optuna one dim
    {'weight_soln_obs': 988.7964820901158,
                       'weight_gap_term': 0.002790913140467998,
                       'lr_soln': 0.0016080738963361809}, # optuna
                      {"weight_soln_obs":966.11,
                       "weight_gap_term":0.000021865,
                       "lr_soln": 0.0066},
                      {"weight_soln_obs":467.34,
                       "weight_gap_term":0.0067495,
                       "lr_soln": 0.0066},]
wdl_space = {"search_space":optuna_search_space,
             "pts":points_to_evaluate}

optuna_lrw_space = {
    #"lr_soln":tune.loguniform(0.0001,0.1),
    #"lr_testfn": tune.loguniform(0.0001,0.1),
    "weight_soln_obs":tune.uniform(50,5000),
    "weight_testfn_obs":tune.uniform(50,500),
}


pts_lrw_space = [
    #{#'lr_soln': 0.002125546453349326,
    # #'lr_testfn': 0.0010008419500359275,
    #    'weight_soln_obs': 5000,"weight_testfn_obs":5000},
    {'weight_soln_obs':500,"weight_testfn_obs":500},
    #{'weight_soln_obs':50000,"weight_testfn_obs":50000},
    {'weight_soln_obs': 1063.0470960768855, 'weight_testfn_obs': 168.70099251164618},]



lrw_space={"search_space":optuna_lrw_space,
         "pts":pts_lrw_space}

optuna_lr_space = {
    "lr_soln":tune.loguniform(0.0001,0.1),
    "lr_testfn": tune.loguniform(0.0001,0.1),
}


pts_lr_space = [
    {'lr_soln':0.003,'lr_testfn':0.0047},
    {'lr_soln':0.0056273266709056,
     'lr_testfn':0.02897486607394667,},
    {'lr_soln': 0.002125546453349326,
     'lr_testfn': 0.0010008419500359275,},
    {'lr_soln': 0.0066,
     'lr_testfn': 0.00013,},
    {'lr_soln': 0.0099,
     'lr_testfn': 0.0005,},
     ]



lr_space={"search_space":optuna_lr_space,
         "pts":pts_lr_space}

optuna_lr_detail_space = {
    "lr_soln":tune.uniform(0.001,0.009),
    "lr_testfn": tune.uniform(0.0001,0.005),
}


pts_lr_detail_space = [
    {'lr_soln':0.0021,
     'lr_testfn':0.001,},
    {'lr_soln': 0.003178173048990878,
     'lr_testfn': 0.004769403326070381,},
     ]



lr_detail_space={"search_space":optuna_lr_detail_space,
         "pts":pts_lr_detail_space}

individual_weight_soln={
    "ns_one_dim":8000,
    "one_dim":8000,
    "CM":8000,
    "sine":1500,
    "MT":500,
    "ns_two_dim":5000,
    "two_dim":1063.0,
    "nochetto_two_dim":5000,}

individual_weight_testfn={
    "ns_one_dim":1500,
    "one_dim":1500,
    "CM":1500,
    "sine":1500,
    "MT":500,
    "ns_two_dim":5000,
    "two_dim":168.70,
    "nochetto_two_dim":5000,}

def train_ray_optuna(cfg: Config) -> None:
    space=lrw_space #lr_space # lrw_space #lr_detail_space #lr_space #lrw_space # wdl_space
    ray.init(dashboard_port = 8097)

    scheduler = MedianStoppingRule(time_attr='training_iteration',
                                   # metric="error_l2",
                                   # mode="min",
                                   grace_period=8,
                                   )
    base_cfg = copy.deepcopy(cfg)
    seed_everything(int(base_cfg.seed))
    algo = OptunaSearch(points_to_evaluate = space["pts"],)
    algo = ConcurrencyLimiter(algo, max_concurrent=10)
    algo = Repeater(algo,repeat=2)
    search_space= space["search_space"]


    base_cfg.use_ray = True
    run = single_ray_train_run(base_cfg)
    
    
    trainable_with_gpu = tune.with_resources(run, {"cpu":1.0,"gpu": 0.10})

    tuner = tune.Tuner(
        trainable_with_gpu,
        tune_config=tune.TuneConfig(
            metric="error_l2",
            mode="min",
            search_alg=algo,
            num_samples=1000, #4
            scheduler=scheduler,
        ),
        run_config=train.RunConfig(stop={"time_total_s": 1200000000},sync_config=train.SyncConfig(sync_artifacts=True),),
        param_space=search_space,
    )
    
    results = tuner.fit()

if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")
    parser.add_argument("--ray_temp_dir", type=str, help="Ray temporary directory.")

    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    cfg = Config(**cfg_dict)
 
    #train_ray_importance(cfg)
    train_ray_grid(cfg, args.ray_temp_dir)
    #train_ray_optuna(cfg)
