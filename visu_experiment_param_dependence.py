import streamlit as st

from pathlib import Path
import json
import pandas as pd

import numpy as np
from scipy.interpolate import griddata
import plotly.figure_factory as ff

import torch
import nnvi.eval_ray_run as eval_r

import altair as alt
import plotly.graph_objects as go
#import pyvista as pv, needs x windos
import nnvi.plot_manager as pm
from streamlit_pdf_viewer import pdf_viewer

st.set_page_config(layout="wide")

@st.cache_data
def get_results(ray_folder,run_name,max_len=5,use_existing_cash=True):
    return eval_r.ray_results(ray_folder,
                              run_name,
                              max_len=max_len,
                              use_existing_cash=use_existing_cash)
@st.cache_data
def get_data(ray_folder,run_name,max_len=5,use_existing_cash=True):
    results = get_results(ray_folder,run_name,max_len,use_existing_cash)
    pdim = results.get_dim()
    exps = results.get_experiments()
    data = []
    for exp_id,experiment in exps.items():
        grid = experiment.get_regular_grid(n_interior=1000,n_boundary=10)
        example = experiment.get_example()
        obstacle = example.obstacle
        soln_nn = experiment.get_soln()
        diff_obs_sol = diff(obstacle,soln_nn)
        fn_data = eval_r.plot_data_for_nn_on_mesh(diff_obs_sol,grid,pdim=pdim,name='diff_obs_sol')
        obs_data = eval_r.plot_data_for_nn_on_mesh(obstacle,grid,pdim=pdim,name='obstacle')
        sol_data = eval_r.plot_data_for_nn_on_mesh(soln_nn,grid,pdim=pdim,name='soln_nn')
        fn_value = fn_data['value']
        touching_points = (np.abs(fn_value)<0.1)
        fraction_touching = np.sum(touching_points)/np.sum(touching_points>-1)
        bc_vals = torch.tensor([[-2],[2]])*1.0
        soln = experiment.get_soln()
        soln_inter = np.mean(np.abs(soln.interpolation(bc_vals).detach().cpu().numpy()))
        soln_f_nn = np.mean(np.abs(soln.f_nn(bc_vals).detach().cpu().numpy()))
        entry={'exp_id':experiment.exp_id,
               'fraction_touching':fraction_touching,
               'loss_b_a_interpol':experiment.final_values_interpolation()["loss_b_after_intrerpol"],
               'loss_obs_a_interpol':experiment.final_values_interpolation()["loss_soln_obs_after_intrerpol"],
               'loss_b_final': experiment.get_final_error("loss_b"),
               'mean_bc_value_at_final': np.mean(torch.abs(soln(bc_vals)).cpu().detach().numpy()),
               'soln_inter':soln_inter,
               'soln_f_nn':soln_f_nn,
               'shift':experiment.get_shift(),
               'obs_data':obs_data,
               'sol_data':sol_data,
               'diff_obs_sol':fn_data,}
        data.append(entry)

    
    return data

def diff(sol,gt):
    if sol is not None and gt is not None:
        def fn(x):
            return sol(x)-gt(x)
    else:
        def fn(x):
            return None
    return fn

def plot_line_data(df,pdim):
    if df is None:
            st.write("No Data")
    if pdim == 1:
        chart = alt.Chart(df).mark_line().encode(
            x='X_x:Q',
            y='value:Q',
            color='name:N'
        )
        st.altair_chart(chart, use_container_width=True)
    elif pdim == 2:
        print('Better use the plot_surface function!!')        
    else:
        raise NotImplementedError


ray_folder = Path("/home/akister/ray_results")
run_name = "fn_2024-05-22_16-34-03"
runs_dict = {
    "fn_2024-05-22_17-33-34": "ns_one_dim with varing shifts, not using freez functions",
    "fn_2024-05-23_15-22-08": "ns_one_dim using the freez_only_interpolation and freez_all functions ",
    "fn_2024-05-23_17-06-44": "ns_one_dim enforce bc freez with deatch()",
    "fn_2024-05-23_17-20-16": "ns_one_dim  xr.requires_grad_(False) in train_step(self, epoch: int), NO deatch()",
    "fn_2024-05-23_17-50-01": "ns_one_dim  restricting the optimizer to the relevant params ,xr.requires_grad_(False) in train_step",
    "fn_2024-05-23_18-02-43": "ns_one_dim just the optimizer restriction, no freez functions!",
    "fn_2024-05-23_18-40-57": "multiple runs per shift, but same setting as fn_2024-05-23_18-02-43",
}
interesting_runs=list(runs_dict.keys())
st.write(runs_dict)

with st.container():
    run_name = st.selectbox(
        'What ray run do you like to analyze?',
        interesting_runs,
        index =1)

    max_len = st.number_input(label='How many experiments should we consider (maximally)?',
                              min_value =1,value = 10, step =1)
    
    results = get_results(ray_folder,run_name,max_len=max_len) 
    pdim = results.get_dim()
    st.write(f'Data of run {run_name} \n Example: {results.get_example().name}')
    with st.expander('The config'):
        config_as_pd=pd.DataFrame.from_dict(results.get_cfg_content(), orient='index')
        st.table(config_as_pd)
    data = get_data(ray_folder,run_name,max_len)
    pd_data=pd.DataFrame(data)
    st.write(pd_data)
    st.scatter_chart(
        pd_data,
        x='shift',
        y='fraction_touching',
        color='mean_bc_value_at_final'
    )

    st.scatter_chart(
        pd_data,
        x='loss_b_a_interpol',
        y='loss_b_final',
    )

    
    idx = st.selectbox(
        'What experiment do you like to look at?',
        range(len(data)),
        index =0)
    data_of_idx = data[idx]
    df = pd.concat([pd.DataFrame(entry) for entry in [data_of_idx['obs_data'],data_of_idx['sol_data']]])
    plot_line_data(df,pdim)
        
    df = pd.concat([pd.DataFrame(entry) for entry in [data_of_idx['diff_obs_sol']]])
    plot_line_data(df,pdim)
        
    experiment = exps[data_of_idx['exp_id']]
    pf_tb = pd.DataFrame(experiment.plots.event_file_content())
    st.write(pf_tb)
    plot_line_data(pf_tb,1)
    st.write(experiment.plots.final_values())
    st.write(experiment.get_final_error("loss_b"))
   # st.write(experiment.plots.get_histor())
