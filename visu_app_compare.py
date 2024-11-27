import streamlit as st

from pathlib import Path
import json
import pandas as pd

import numpy as np
from scipy.interpolate import griddata
import plotly.figure_factory as ff

import nnvi.eval_ray_run as eval_r

import altair as alt
import plotly.graph_objects as go

from visu_app import interesting_runs, get_results,diff

runs_with_zero_gt = ["fn_2024-05-08_08-40-54"]

ray_folder = Path("/home/akister/ray_results")

with st.container():
    use_zero_gt=st.checkbox(
        "Do you like to compare to runs with zeor gt?",
        value = True)
    run_name_A = st.selectbox(
        'Choose ray run A:',
        interesting_runs,
        index = 3)
    if use_zero_gt:
        runs_for_B = runs_with_zero_gt
    else:
        runs_for_B = interesting_runs
    run_name_B = st.selectbox(
        'Choose ray run B:',
        runs_for_B,
        index =0)

    metric_for_choosing_the_best = st.selectbox(
        'Choos the metric for choosing the best experiment',
        eval_r.possible_metrics)

    max_len = st.number_input(label='How many experiments should we consider (maximally)?',
                              min_value =1,value = 10, step =1,key=1)
    
    compar = {'A': {'results': get_results(ray_folder,
                                           run_name_A,
                                           max_len=max_len)},
              'B': {'results': get_results(ray_folder,
                                           run_name_B,
                                           max_len=max_len)} 
    }
    
    
    grid= compar['A']['results'].get_regular_grid(n_interior=1000,
                                                   n_boundary=1000)
    pdim = compar['A']['results'].get_dim()
    
    for k,v in compar.items():
        v['exp']=v['results'].get_best_experiment(metric_for_choosing_the_best)
        v['fn']=v['exp'].get_soln()
    
    compar['diff'] = {}
    compar['diff']['fn'] = diff(compar['A']['fn'],compar['B']['fn'])
    for name,v in compar.items():
        v['data']=eval_r.plot_data_for_nn_on_mesh(
            v['fn'],
            grid,
            pdim,
            name=name)
        fn_data = v['data']
        x = fn_data['X_x']
        y = fn_data['X_y']
        z = fn_data['value']
        
        trace = go.Surface(x=x, y=y, z=z, name=name)
        fig = go.Figure(data=trace)
        fig.update_layout(
            title=f'{name}',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title=name
            )
        )
        v['fig']=fig
    cols = st.columns(2)
    with cols[0]:
        st.plotly_chart(compar['A']['fig'])
    with cols[1]:
        st.plotly_chart(compar['B']['fig'])
    st.plotly_chart(compar['diff']['fig'])
        
