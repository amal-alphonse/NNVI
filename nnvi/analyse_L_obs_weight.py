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


import nnvi as root_module
from torch import Tensor, optim


from nnvi.grids.domain import Domain
from nnvi.config import Config
from nnvi.examples import EXAMPLES_MAP, Example
from nnvi.loss import BaseLoss, build_loss

#from nnvi.grids.grid import * 
from nnvi.grids.grid import Grid, ImportanceForSolution, ImportanceForTest

from nnvi.metrics import L2Error, LInftyError, FreeBcSize,Metric, Visualise
from nnvi.models import build_model

from nnvi.callbacks import Callback, SolutionGradient, TensorboardLogger, Tunes, CheckpointLogger
from nnvi.grids.grid import SimpleGrid
from nnvi.train import Trainer


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

chin_search_space={
    "weight_soln_obs": tune.uniform(100,1e5),
    "weight_testfn_obs":tune.uniform(100,1e5),
}

my_search_space={
    "weight_soln_obs": tune.uniform(3900,5000),
    "weight_testfn_obs":tune.uniform(3900,5000),
}
points_to_evaluate_for_one_dim=[
    {"weight_soln_obs":750,"weight_testfn_obs":750},
    {"weight_soln_obs":1000,"weight_testfn_obs":1000},
    {"weight_soln_obs":2000,"weight_testfn_obs":2000},
    {"weight_soln_obs":3000,"weight_testfn_obs":3000},
    {"weight_soln_obs":4000,"weight_testfn_obs":4000},
    {"weight_soln_obs":5000,"weight_testfn_obs":5000},
    {"weight_soln_obs":6000,"weight_testfn_obs":6000},
    {"weight_soln_obs":7500,"weight_testfn_obs":7500},
    {"weight_soln_obs":10000,"weight_testfn_obs":10000},
    {"weight_soln_obs":50000,"weight_testfn_obs":50000},]

points_for_search = [
    {"weight_soln_obs":4000,"weight_testfn_obs":4000},
    {"weight_soln_obs":4500,"weight_testfn_obs":4500},
    {"weight_soln_obs":5000,"weight_testfn_obs":5000},]

for_analyis = {'space':chin_search_space, 'pts':points_to_evaluate_for_one_dim,'num_samples':40}
for_search = {'space':my_search_space, 'pts':points_for_search,'num_samples':400}
setting = for_search#for_analyis
def train_ray_importance(cfg: Config) -> None:
    ray.init(dashboard_port = 8097)
    base_cfg = copy.deepcopy(cfg)
    seed_everything(int(base_cfg.seed))
    algo = BayesOptSearch(utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0},
                          points_to_evaluate = setting['pts'],
                          random_search_steps=10,
                          verbose = 1,
                          patience = 2000,
                          skip_duplicate = False)
    algo = Repeater(algo,repeat=4)
    search_space=setting['space'] #chin_search_space    

    def single_ray_train_run(cfg):
        def fn(ray_cfg):
            if "shift" in ray_cfg:
                shift = ray_cfg["shift"]
                ray_cfg.pop("shift", None)
            else:
                shift = None
            copy_of_cfg=copy.deepcopy(cfg)
            ray_cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            copy_of_cfg = fusion_config_dict_ray_dict(copy_of_cfg,ray_cfg)
            
            
            if isinstance(copy_of_cfg.soln_model_args['width'],tuple):
                copy_of_cfg.soln_model_args['width']=copy_of_cfg.soln_model_args['width'][0]
                copy_of_cfg.soln_model_args['depth']=copy_of_cfg.soln_model_args['depth'][0]
                copy_of_cfg.soln_model_args['hc_bias'] = copy_of_cfg.soln_model_args['hc_bias'][0]
                copy_of_cfg.testfn_model_args['hc_bias'] = copy_of_cfg.testfn_model_args['hc_bias'][0]
            
            seed_dict = {"seed":np.random.randint(low=2,high=99999)}
            copy_of_cfg = replace(copy_of_cfg,**seed_dict)
            
            check_folder_dict ={"chpt_folder": str(os.getcwd())}
            copy_of_cfg = replace(copy_of_cfg,**check_folder_dict)

            save_cfg_to = Path(copy_of_cfg.chpt_folder)/"cfg.json"
            with save_cfg_to.open("w", encoding="UTF-8") as target: 
                json.dump(asdict(copy_of_cfg), target)
            
            example = EXAMPLES_MAP[copy_of_cfg.example]
            if copy_of_cfg.example == "ns_one_dim" and shift is not None:
                example.set_shift_obstacle(shift)
                save_example_setting = Path(copy_of_cfg.chpt_folder)/"example_setting.json"
                example_setting = {'shift':shift}
                with save_example_setting.open("w", encoding="UTF-8") as target: 
                    json.dump(example_setting, target)

            soln_model = build_model(
                copy_of_cfg.soln_model_name,
                in_N=example.domain.pdim,
                out_N=1,
                force_bc=cfg.force_bc,
                example=example,
                name_of_example=copy_of_cfg.example,
                **copy_of_cfg.soln_model_args,
            )
            testfn_model = build_model(
                copy_of_cfg.testfn_model_name,
                in_N=example.domain.pdim,
                out_N=1,
                example=example,
                force_bc=cfg.force_bc,
                name_of_example=copy_of_cfg.example,
                **copy_of_cfg.testfn_model_args,
            )
            loss = build_loss(example, copy_of_cfg,integration_type="weighted_mc")
            grid = getattr(root_module.grids.grid, copy_of_cfg.grid_cls)(example.domain, copy_of_cfg.n_interior, copy_of_cfg.n_boundary)
            grid_for_eval = getattr(root_module.grids.grid, copy_of_cfg.grid_cls)(example.domain, 10000, copy_of_cfg.n_boundary)
            
            board_chapt = Path(os.getcwd()+'/board/')
            board_chapt.mkdir(exist_ok=True,parents=True)
            chkpt_chapt = Path(os.getcwd()+'/check/')
            chkpt_chapt.mkdir(exist_ok=True,parents=True)
            callbacks = [Tunes(best_l2=10.81),
                         TensorboardLogger(str(board_chapt),add_time_stamp=False),
                         CheckpointLogger(str(chkpt_chapt),epochs=copy_of_cfg.epochs,nbr_of_intermediate_chepts=12)]
            
            metrics = [L2Error(), LInftyError(),FreeBcSize(),Visualise(example,use_trisurf=True)]
            
            trainer = Trainer(
                copy_of_cfg, example, 
                grid, soln_model, 
                testfn_model, loss, 
                metrics, callbacks)
            v = trainer.train()
        return fn
    
    run = single_ray_train_run(base_cfg)
    
    
    trainable_with_gpu = tune.with_resources(run, {"cpu":1.0,"gpu": 0.25})

    tuner = tune.Tuner(
        trainable_with_gpu,
        tune_config=tune.TuneConfig(
            metric="error_l2", #"best_l2"
            mode="min",
            search_alg=algo,
            num_samples=setting['num_samples'],
        ),
        run_config=train.RunConfig(stop={"time_total_s": 12000}),
        param_space=search_space,
    )
    
    results = tuner.fit()



if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")

    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    cfg = Config(**cfg_dict)
    cfg_update={"seed":cfg.seed}
    cfg=replace(cfg,**cfg_update)
 
    train_ray_importance(cfg)
    
