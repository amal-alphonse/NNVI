from typing import Any, Optional

import torch
import torch.nn as nn
import nnvi as root_module
from nnvi.models import build_model

from dataclasses import dataclass, field, replace, asdict

from nnvi.config import Config
from nnvi.examples import EXAMPLES_MAP, Example
from nnvi.grids.grid import Grid, ImportanceForSolution, ImportanceForTest
from nnvi.grids.grid import SimpleGrid

from nnvi.loss import BaseLoss, build_loss

import nnvi as root_module

from pathlib import Path
import json
import pandas as pd
import numpy as np

import pickle


ray_folder = Path("/home/akister/ray_results")
run_name = "fn_2024-04-25_15-38-17"#"fn_2024-04-24_15-03-26"#"fn_2024-04-15_09-27-29"

class experiment_state():
    def __init__(self,experiment_state):
        self.experiment_state = experiment_state

class searcher_state():
    def __init__(self,searcher_state):
        self.searcher_state = searcher_state

class search_gen_state():
    def __init__(self,search_gen_state):
        self.search_gen_state = search_gen_state

class tuner():
    def __init__(self,tuner):
        self.tuner = tuner

def ray_files_adresses(ray_folder,run_name):
    folder = ray_folder / run_name
    file_names = ["experiment_state", "searcher-state", "search_gen_state","tuner"]
    file_dict={}
    for fn in file_names:
        file_dict[fn]=list(folder.glob(fn+"*"))[0]
    return file_dict

def exp_name2id(exp_name):
    split = exp_name.split('_')
    if ('example' in exp_name) or ('obs' in exp_name) or ('diagonal' in exp_name) or ('weight' in exp_name):
        exp_id = split[2]
    else:
        exp_id = split[1]
    return exp_id

class params():
    def __init__(self,params_json,params_pkl):
        self.params_json = params_json # json & pkl contain the same info
        self.params_pkl = params_pkl
        if self.params_pkl is not None:
            with open(str(self.params_pkl),'rb') as f:
                self.params_pkl_data =  pickle.load(f)
        if self.params_json is not None:
            with open(str(self.params_json),'r') as f:
                data = f.read().replace('\n', '')
                self.params_json_data = json.loads(data)
def name2model(name):
    if name == "DrrnnWithSpecificBC":
        model_name = "DrrnnWithSpecificBC"
    elif name == "NewModelFreeBCclassifier":
        model_name = "NewModelFreeBCclassifierPlain"
    else:
        model_name = name
    return model_name
class ExampleSetting():
    def __init__(self,example_setting):
        self.example_setting = example_setting
        if self.example_setting is not None:
            with open(str(self.example_setting),'r') as f:
                data = f.read().replace('\n', '')
                self.example_setting_data = json.loads(data)
    def get_shift(self):
        if self.example_setting_data['shift'] is None:
            shift = 0
        else:
            shift = self.example_setting_data['shift']
        return shift

class cfg():
    def __init__(self,cfg,example_setting=None):
        self.cfg=cfg
        if example_setting is None:
            self.example_setting = None
        else:
            self.example_setting = ExampleSetting(example_setting)

        if self.cfg is not None:
            with open(str(self.cfg),'r') as f:
                data = f.read().replace('\n', '')
                self.cfg_data = json.loads(data)
            self.cfg_obj = Config.from_dict(self.cfg_data)
        else:
            self.cfg_data = None
            self.cfg_obj = None
        self.example = None
    def get_example(self):
        if self.example is None:
            self.example = EXAMPLES_MAP[self.cfg_obj.example]
            if self.cfg_obj.example == "ns_one_dim":
                if self.example_setting is not None:
                    self.example.set_shift_obstacle(self.example_setting.get_shift())
        return self.example
    def get_shift(self):
        if self.cfg_obj.example == "ns_one_dim":
            shift = self.example_setting.get_shift()
        else:
            shift = 0
        return shift
    def get_soln_model(self):
        example = self.get_example()
        soln_model = build_model(
        name2model(self.cfg_obj.soln_model_name),
        in_N=example.domain.pdim,
        out_N=1,
        force_bc=self.cfg_obj.force_bc,
        example=example,
        name_of_example=self.cfg_obj.example,
        pretrained=True,
        **self.cfg_obj.soln_model_args,
        )
        return soln_model
    def get_test_model(self):
        example = self.get_example()
        testfn_model = build_model(
        name2model(self.cfg_obj.testfn_model_name),
        in_N=example.domain.pdim,
        out_N=1,
        force_bc=self.cfg_obj.force_bc,
        example=example,
        name_of_example=self.cfg_obj.example,
        pretrained=True,
        **self.cfg_obj.testfn_model_args,
        )
        return testfn_model
    def get_grid(self,n_interior=1000,n_boundary=1000):
        example = self.get_example()
        grid = getattr(root_module.grids.grid, self.cfg_obj.grid_cls)(example.domain, 
                                                                      n_interior, 
                                                                      n_boundary)
        return grid
    def get_regular_grid(self,n_interior=1000,n_boundary=1000):
        example = self.get_example()
        grid = getattr(root_module.grids.grid, "SimpleGrid")(example.domain, 
                                                                      n_interior, 
                                                                      n_boundary,
                                                                      randomise=False)
        return grid
    def get_loss(self):
        example = self.get_example()
        loss = build_loss(example, self.cfg_obj,integration_type="weighted_mc")
        return loss

    def get_cfg_content(self):
        return asdict(self.cfg_obj)

    def get_cfg_object(self):
        return self.cfg_obj
        
    def get_dim(self):
        return self.get_example().domain.pdim
    
    def get_obs_loss(self):
        return self.cfg_obj.weight_soln_obs
    

class progress():
    def __init__(self,progress_csv):
        self.progress_csv = progress_csv
    def get_data(self):
        return pd.read_csv(self.progress_csv)

possible_metrics = [ 'loss_r','loss1', 'loss2',  'error_linfty', 'best_l2',  'current_l2', 'loss_b', 'loss_soln_obs', 'loss_testfn_obs', 'weighted_loss_b', 'weighted_loss_soln_obs', 'weighted_loss_testfn_obs', 'pre-weighted_loss2', 'loss_test_b', 'weighted_loss_test_b', 'distance_to_diagonal', 'error_l2',  'timestamp', 'training_iteration',  'time_this_iter_s', 'time_total_s']
class result():
    def __init__(self,result_json):
        self.result_json = result_json
        self.result_data = None
        
    def get_data(self):
        if self.result_data is None:
            if self.result_json is not None:
                self.result_data = []
                with open(str(self.result_json)) as f:
                    for line in f:
                        self.result_data.append(json.loads(line))
            else:
                self.result_data = None
        return self.result_data

    def get_final_error(self,metric='best_l2'):
        """
        possible values for metric are given in the list possible_metrics above
        """
        data = self.get_data()
        f_e = None
        if data is None:
            f_e = None
        else:
            if metric in data[-1].keys():
                last_idx=-1
            elif metric in data[-2].keys():
                last_idx=-2
            else:
                last_idx=-3 
            f_e = data[last_idx][metric] 
        return f_e

    def get_initial_error(self,metric='best_l2'):
        """
        possible values for metric are given in the list possible_metrics above
        """
        data = self.get_data()
        f_e = None
        if data is None:
            f_e = None
        else:
            f_e = data[0][metric] 
        return f_e

from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import tensorboard as tb

class plots():
    def __init__(self,plots):
        self.data = None
        if plots is None:
            self.plots = None
            self.tb_event = None
        else:
            self.plots = Path(plots)/"interpolated_functions"
            self.tb_event = list(self.plots.rglob("events.out.tfevents*"))[0]
    def event_file_content(self):
        data=[]
        if self.plots is not None:
             # Use EventAccumulator to read the TensorBoard file
            event_acc = EventAccumulator(str(self.tb_event))
            event_acc.Reload()
            
            
            # Iterate through the scalar events
            for tag in event_acc.Tags()['scalars']:  # Focus on scalar summaries
                for event in event_acc.Scalars(tag):
                    data.append({'name':tag,'X_x':event.step, 'value':event.value}) 
        self.data = pd.DataFrame(data)
        return data
    
    def final_v(self,name='loss_b'):
        if self.data is None:
            self.event_file_content()
        sorted_pd = self.data[self.data["name"] == name].sort_values(by='X_x')
        if len(np.array(sorted_pd["value"]))>0:
            v = np.array(sorted_pd["value"])[-1]
        else:
            v = None
        return v
    def final_values(self):
        names = ["loss_b","loss_soln_obs"]
        v_s = {}
        for name in names:
            v_s[name+"_after_intrerpol"]=self.final_v(name)
        return v_s

def model_from_chpt(chpt,model):
    model.load_state_dict(torch.load(chpt))
    model.eval()
    return model

class Check():
    def __init__(self,check,cfg=None):
        self.check=check
        self.cfg=cfg
        if self.check is not None:
            self.check_soln = check/'soln.pt'
            self.check_test = check/'test.pt'
        else:
            self.check_soln = None
            self.check_test = None
    def get_soln(self):
        model = None
        if self.check_soln is not None and self.cfg is not None:
            if Path(self.check_soln).exists(): 
                model = self.cfg.get_soln_model()
                model = model_from_chpt(self.check_soln,model)
        return model
    def get_test(self):
        model = None
        if self.check_test is not None and self.cfg is not None:
            if Path(self.check_test).exists() :
                model = self.cfg.get_test_model()
                model = model_from_chpt(self.check_test,model)
        return model

class Check_hist():
    def __init__(self,check,cfg=None):
        self.check = check
        if check is not None:
            self.hist = [{'epoch':f.name,'chpt':Check(f,cfg)} for f in self.check.iterdir() if f.is_dir()]
        else:
            self.hist = None
    def get_all_checkpts(self):
        return self.hist
    def get_last_soln(self):
        if self.hist == None:
            model = None
        else:
            for hi in reversed(self.hist):
                model = hi['chpt'].get_soln()
                if model is not None:
                    break
        return model

    def get_last_test(self):
        if self.hist == None:
            model = None
        else:
            for hi in reversed(self.hist):
                model = hi['chpt'].get_test()
                if model is not None:
                    break
        return model

def exp_file_adresses(path):
    folder = path
    file_names = ["params.json", "params.pkl", "progress.csv", "result.json","interpolants","plots","events.out.tfevents","check","cfg.json","example_setting.json"]
    file_dict={}
    for fn in file_names:
        candidates = list(folder.glob(fn+"*"))
        if len(candidates)>0:
            adress = candidates[0]
        else:
            adress = None
        file_dict[fn]=adress
    return file_dict
    
class ray_experiment():
    def __init__(self,path,):
        self.path = path
        self.exp_name = self.path.name
        self.exp_id = exp_name2id(self.exp_name)
        self.exp_files = exp_file_adresses(path)
        self.params=params(self.exp_files["params.json"],self.exp_files["params.pkl"])
        self.progress=progress(self.exp_files["progress.csv"])
        self.result=result(self.exp_files["result.json"])
        self.cfg = cfg(self.exp_files["cfg.json"],self.exp_files["example_setting.json"])
        self.check = Check(self.exp_files["check"],self.cfg)
        self.check_hist = Check_hist(self.exp_files["check"],self.cfg)
        self.plots = plots(self.exp_files["plots"])
        
    def final_values_interpolation(self):
        return self.plots.final_values()
    def get_shift(self):
        return self.cfg.get_shift()

    def get_obs_loss(self):
        return self.cfg.get_obs_loss()

    def get_nbr(self):
        return self.exp_name.split('_')[2]
    def get_final_error(self,metric='best_l2'):
        return self.result.get_final_error(metric=metric)

    def get_initial_error(self,metric='best_l2'):
        return self.result.get_initial_error(metric=metric)

    def get_soln(self):
        model = self.check.get_soln() 
        if model is None:
            model = self.check_hist.get_last_soln()
        return model
    def get_test(self):
        model = self.check.get_test()
        if model is None:
            model = self.check_hist.get_last_test()
        return model
    def get_grid(self,n_interior=1000,n_boundary=1000):
        return self.cfg.get_grid(n_interior,n_boundary)
    def get_regular_grid(self,n_interior=1000,n_boundary=1000):
        return self.cfg.get_regular_grid(n_interior,n_boundary)
    def get_loss(self):
        return self.cfg.get_loss()
    def get_example(self):
        return self.cfg.get_example()
    def get_example_name(self):
        return self.get_example().name
    def get_cfg_content(self):
        return self.cfg.get_cfg_content()
    def get_cfg_object(self):
        return self.cfg.get_cfg_object()
    def get_diag_force(self):
        if "diagonal_forcing" in self.get_cfg_content().keys():
            r = self.get_cfg_content()["diagonal_forcing"]
        else:
            r = self.get_cfg_content()["weight_gap_term"]
        return r
    def get_dim(self):
        return self.cfg.get_dim()
    def get_training_process(self):
        return self.progress.get_data()
    def get_check_hist(self):
        return self.check_hist.hist

def ray_experiments_by_folder(folder):
    experiments = {}
    for f in folder.glob("fn_*"):
        ray_exp_object = ray_experiment(f)
        experiments[ray_exp_object.exp_id]=ray_exp_object
    return experiments

def ray_experiments(ray_folder,run_name):
    folder = ray_folder / run_name
    return ray_experiments_by_folder(folder)


        
class ray_set_of_experiments():
    def __init__(self,set_of_experiments):
        self.ray_experiments = set_of_experiments
        self.max_len = None
        self.searched_for_a_experiment = False
        self.a_experiment = None
        self.short_exps = None

    def get_experiments(self):
        if self.short_exps is None:
            exp_s={}
            if self.max_len is not None:
                for i,(k,v) in enumerate(self.ray_experiments.items()):
                    exp_s[k]=v
                    if i>self.max_len:
                        break
            else:
                exp_s = self.ray_experiments
            self.short_exps = exp_s
        return self.short_exps
    
    def get_final_errors(self,metric='best_l2'):
        exp_s=self.get_experiments()
        final_errors={}
        for k,v in exp_s.items():
            f_e=v.get_final_error(metric=metric)
            final_errors[k]=f_e
        return final_errors

    def get_best_experiment(self,metric='best_l2'):
        errors = self.get_final_errors(metric)
        best_exp_id=min(errors,key=errors.get)
        return self.get_experiments()[best_exp_id]
    def get_initial_errors(self,metric='best_l2'):
        exp_s=self.get_experiments()
        final_errors={}
        for k,v in exp_s.items():
            f_e=v.get_initial_error(metric=metric)
            final_errors[k]=f_e
        return final_errors

    def get_a_complet_experiment(self):
        if self.searched_for_a_experiment is False:
            for k,v in self.ray_experiments.items():
                if v.cfg is not None:
                    self.a_experiment = v
                    break
            self.searched_for_a_experiment = True
        return self.a_experiment
        
    def get_grid(self,n_interior=1000,n_boundary=1000):
        experiment = self.get_a_complet_experiment()
        return experiment.get_grid(n_interior,n_boundary)

    def get_regular_grid(self,n_interior=1000,n_boundary=1000):
        experiment = self.get_a_complet_experiment()
        return experiment.get_regular_grid(n_interior,n_boundary)
    
    def get_example(self):
        experiment = self.get_a_complet_experiment()
        return experiment.get_example()
    def get_example_name(self):
        return self.get_example().name
    def get_loss(self):
        experiment = self.get_a_complet_experiment()
        return experiment.get_loss()
    def get_cfg_content(self):
        experiment = self.get_a_complet_experiment()
        return experiment.get_cfg_content()
    def get_cfg_object(self):
        experiment = self.get_a_complet_experiment()
        return experiment.get_cfg_object()
    def get_dim(self):
        experiment = self.get_a_complet_experiment()
        return experiment.get_dim()
    
class ray_results(ray_set_of_experiments):
    def __init__(self,ray_folder,run_name,max_len=None,use_existing_cash=True):
        self.ray_folder = ray_folder
        self.run_name = run_name
        self.file_dict = ray_files_adresses(ray_folder,run_name)
        super().__init__( ray_experiments(ray_folder,run_name))
        self.max_len = max_len

        self.cash=sol_test_matrix_cash(self.ray_folder,self.run_name)
        if not use_existing_cash:
            self.cash.clean_cash()
        self.searched_for_a_experiment = False
        self.a_experiment = None
        self.short_exps = None

    def clean_sol_test_matrix_cash(self):
        self.cash.clean_cash()
    
    def set_max_len(self,max_len=None):
        self.max_len = max_len
        self.short_exps = None

    
    
    def get_experiment_by_nbr(self,exp_nbr):
        exp = None
        for (k,v) in self.ray_experiments.items():
            if v.get_nbr() == exp_nbr:
                exp = v
        return exp
        
    

        
    def loss_r_matrix(self,nb_gird_pts=100000,empty_cash=False):
        results=[]
        loss = self.get_loss()
        grid = self.get_grid(nb_gird_pts,10)
        example = self.get_example()
        es = self.get_experiments()
        for i_a,e_a in enumerate(es.values()):
            soln = e_a.get_soln()
            for i_b,e_b in enumerate(es.values()):
                if not self.cash.entry_exists(sol_id=e_a.exp_id,test_id=e_b.exp_id):
                    test = e_b.get_test()
                    if (soln is not None) and (test is not None):
                        loss_v = apply_loss(loss=loss,
                                            grid=grid,
                                            soln=soln,
                                            test=test,
                                            example=example,
                                            device="cuda")
                        loss_r = float(loss_v['loss_r'].detach().cpu().numpy())
                        loss1 = float(loss_v['loss1'].detach().cpu().numpy())
                    else:
                        loss_r =  None
                    entry = {'sol_id':e_a.exp_id,
                             'test_id':e_b.exp_id,
                             'loss_r': loss_r,
                             #'loss1': loss1
                         }
                    results.append(entry)
                    self.cash.write_entries([entry],overwrite=False)
                    del test
                    del entry
                else:
                    results.append(self.cash.get_entry(sol_id=e_a.exp_id,test_id=e_b.exp_id,cash_format=False))
                if self.max_len is not None and i_b>self.max_len:
                    break
            del soln
            if self.max_len is not None and i_a>self.max_len:
                break
        return results

def group_by_example(ray_experiments):
    grouped={}
    examples=set([e.get_example_name() for e in ray_experiments.values()])
    for example in examples:
        grouped[example]={}
    for exp_id,experiment in ray_experiments.items():
        grouped[experiment.get_example_name()][exp_id]=experiment
    return {k:ray_set_of_experiments(v) for k,v in grouped.items()}

def group_by_diag_force(ray_experiments):
    grouped={}
    forces=set([e.get_diag_force() for e in ray_experiments.values()])
    for force in forces:
        grouped[force]={}
    for exp_id,experiment in ray_experiments.items():
        grouped[experiment.get_diag_force()][exp_id]=experiment
    return {k:ray_set_of_experiments(v) for k,v in grouped.items()}

class ray_grid():
    def __init__(self,folder):
        self.folder = folder
        self.ray_experiments = ray_experiments_by_folder(folder)
        self.grouped_by_example = group_by_example(self.ray_experiments)
        self.group_by_diag_force = group_by_diag_force(self.ray_experiments)
    def get_example_names(self):
        return self.grouped_by_example.keys()
    def get_results_for_example(self,example_name='CM'):
        return self.grouped_by_example[example_name]
    def get_possible_diag_forces(self):
        return self.group_by_diag_force.keys()
    def get_results_for_diag_force(self,diag_force):
        return self.group_by_diag_force[diag_force]
        
def apply_loss(loss,grid,soln,test,example,device="cuda"):
    soln.train().to(device)
    test.train().to(device)
    
    xr = grid.get_interior_points().to(device)
    xb = grid.get_boundary_points().to(device)
    xr.requires_grad_()
    
    f_value = example.source_term(xr)
    soln_interior = soln(xr)
    soln_boundary = soln(xb)
    testfn_interior = test(xr)
    testfn_boundary = test(xb)

    losses = loss(
        xr,
        xb,
        soln_interior,
        soln_boundary,
        testfn_interior,
        f_value,
        testfn_boundary=testfn_boundary,
        epoch=100000000000,
    )

    return losses





def compare_ids(d,sol_id,test_id):
    equale = False
    if d['sol_id'] == sol_id and d['test_id'] == test_id:
        equale = True
    return equale




        


def cash_naming_convention(entry):
    if 'sol_id' in entry.keys():
        sol_id = entry['sol_id']
    else:
        sol_id = entry['soln']
        entry.pop('soln',None)
    if 'test_id' in entry.keys():
        test_id = entry['test_id']
    else:
        test_id = entry['test']
        entry.pop('test', None)
    entry['sol_id'] = sol_id
    entry['test_id'] = test_id
    return entry

class sol_test_matrix_cash():
    def __init__(self,ray_folder,run_name):
        """
        For cashing sol test comparisons
        """
        self.cash_adress = Path(ray_folder)/run_name / "sol_test_matrix_cash.jsonl"
        self.existing_sol_ids = None
        self.existing_test_ids = None
        
    def get_data(self):
        if self.cash_adress.exists():
            with open(self.cash_adress, "r") as f:
                data = [json.loads(line.strip()) for line in f]
        else:
            data = []
        return data

    def get_entry(self,sol_id,test_id,cash_format=False):
        hits = [d  for d in self.get_data() if compare_ids(d,sol_id,test_id)]
        if len(hits)>0:
            v = hits[0]
        else:
            v = {'sol_id':sol_id,
                 'test_id':test_id,
                 'loss_r': None,}
        return v
    def get_existing_sol_ids(self):
        if self.existing_sol_ids is None:
            self.existing_sol_ids = list(set([d['sol_id'] for d in self.get_data()]))
        return self.existing_sol_ids

    def get_existing_test_ids(self):
        if self.existing_test_ids is None:
            self.existing_test_ids = list(set([d['test_id'] for d in self.get_data()]))
        return self.existing_test_ids

    def entry_exists(self,sol_id,test_id):
        existing_sol_ids = self.get_existing_sol_ids()
        existing_test_ids = self.get_existing_test_ids()
        if existing_sol_ids is not None and existing_test_ids is not None:
            exists = sol_id in existing_sol_ids and test_id in existing_test_ids
        else:
            exists = False
        return exists

    def adopte_naming_conventions(self,list_of_entries):
        return [cash_naming_convention(entry) for entry in list_of_entries]
        
    def remove_existing_entries(self,list_of_entries):
        list_of_entries=self.adopte_naming_conventions(list_of_entries)
        return [entry for entry in list_of_entries if (not self.entry_exists(entry['sol_id'],entry['test_id']))]
        
    def write_entries(self,list_of_entries,overwrite=False):
        """
        list_of_entries=[{'sol_id': ... ,'test_id': ... ,'value': ...},{}]
        """
        list_of_entries = self.adopte_naming_conventions(list_of_entries)
        if overwrite == False:
            clean_list = self.remove_existing_entries(list_of_entries)
        else:
            clean_list = list_of_entries
            breakpoint() # Not implemented
        new_data = []
        for entry in clean_list:
            new_data.append(entry)
        with open(self.cash_adress, "a") as file:
            file.writelines(json.dumps(entry) + "\n" for entry in new_data)
        self.existing_test_ids = None
        self.existing_sol_ids = None
        
    def clean_cash(self):
        if self.cash_adress.exists():
            self.cash_adress.unlink()


def plot_data_for_nn(nn,grid,pdim,device="cpu",name=None):
    if nn is None or grid is None:
        return None
    data={}
    #nn.to(device)
    xr = grid.get_interior_points().to(device)
    xb = grid.get_boundary_points().to(device)
    
    nn_interior = nn(xr)
    nn_boundary = nn(xb)
    
    data['value'] = list(torch.cat((nn_interior.detach(),nn_boundary.detach()),0).to('cpu').numpy().flatten())
    
    x_pt_list = torch.cat((xr,xb),0).to('cpu').numpy()
    if pdim == 1:
        data['X_x'] = list(x_pt_list.flatten())
    elif pdim == 2:
        data['X_x'] = list(x_pt_list[:,0])
        data['X_y'] = list(x_pt_list[:,1])
    else:
        breakpoint()
    if name is not None:
        data['name']=[name]*len(data['value'])
    
    return pd.DataFrame(data)

def plot_data_for_nn_on_mesh(nn,grid,pdim,device="cpu",name=None):
    if nn is None or grid is None:
        return None
    if isinstance(nn, torch.nn.Module):
        nn.to(device)
    data={} 
    X,Y,xr = grid.get_mesh_points()
    
    nn_interior = nn(xr.to(device))
    
    x_pt_list = xr.to('cpu').numpy()
    if pdim == 1:
        data['X_x'] = list(x_pt_list.flatten())
        data['value'] = list(nn_interior.detach().to('cpu').numpy().flatten())
    elif pdim == 2:
        data['X_x'] = X#list(x_pt_list[:,0])
        data['X_y'] = Y#list(x_pt_list[:,1])
        data['value'] = (nn_interior.detach()).to('cpu').numpy().reshape(X.shape)
    else:
        breakpoint()
    if name is not None:
        data['name']=[name]*len(data['value'])
    
    return data

"""
results = ray_results(ray_folder,run_name,max_len=50)
first_exp_name = list(results.ray_experiments.keys())[0]
first_exp = results.ray_experiments[first_exp_name]

m = first_exp.cfg.get_soln_model()
m_v_s = first_exp.get_soln()
m_v_t = first_exp.get_test()
example = first_exp.get_example()
grid = first_exp.get_grid()
loss = first_exp.get_loss()
loss_v = apply_loss(loss=loss,
                    grid=grid,
                    soln=m_v_s,
                    test=m_v_t,
                    example=example
                    ,device="cuda")
loss_r_s = loss_r_matrix(results)
data_set = pd.DataFrame(loss_r_s)
worst_case_test=data_set.groupby(['soln']).loss_r.agg('max').reset_index().set_index('soln')
best_case_test=data_set.groupby(['soln']).loss_r.agg('min').reset_index().set_index('soln')
# worst_case_test.idxmin()
f_e_s = results.get_final_errors(metric='best_l2')#'current_l2'
f_e_d_s = pd.DataFrame([{'soln':k, 'l2':v} for k,v in f_e_s.items()]).set_index('soln')
compare_l2_to_worst=worst_case_test.join(f_e_d_s).sort_values(by='loss_r')
compare_l2_to_best=best_case_test.join(f_e_d_s).sort_values(by='loss_r')
breakpoint()
"""
