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
#import pyvista as pv, needs x windos
import nnvi.plot_manager as pm
from streamlit_pdf_viewer import pdf_viewer

st.set_page_config(layout="wide")
def diff(sol,gt):
    if sol is not None and gt is not None:
        def fn(x):
            return sol(x)-gt(x)
    else:
        def fn(x):
            return None
    return fn

def line_plot(df, x_col, y_cols):
    line = alt.Chart(df).mark_line().encode(
        x=x_col,
        y=y_cols,
        color=alt.Color(alt.repeat('row'), type='nominal')
    ).repeat(row=y_cols)
    return line

@st.cache_data
def get_results(ray_folder,run_name,max_len=5,use_existing_cash=True):
    return eval_r.ray_results(ray_folder,
                              run_name,
                              max_len=max_len,
                              use_existing_cash=use_existing_cash)

@st.cache_data
def get_loss_matrix_soln_test(ray_folder,run_name,max_len=5,use_existing_cash=True):
    results_for_matrix = get_results(ray_folder,
                                     run_name,
                                     max_len,
                                     use_existing_cash=use_existing_cash)#eval_r.ray_results(ray_folder,run_name,max_len=max_len)
    return pd.DataFrame(results_for_matrix.loss_r_matrix())

@st.cache_data
def get_matrix_metric_vs_performanc(ray_folder,run_name,max_len,metric_for_indicator,metric_for_performance,use_existing_cash=True):
    loss_matrix_soln_test = get_loss_matrix_soln_test(ray_folder,
                                                      run_name,
                                                      max_len,
                                                      use_existing_cash)
    data_indicator = loss_matrix_soln_test.groupby(['sol_id']).loss_r.agg(metric_for_indicator).reset_index().set_index('sol_id')
    results_for_matrix = get_results(ray_folder,run_name,max_len)
    data_performanc = pd.DataFrame(
        [{'exp_id':k,metric_for_performance:v} for k,v in  results_for_matrix.get_final_errors(metric=metric_for_performance).items()]
    ).set_index('exp_id')
    compare_indicator_performance = data_indicator.join(data_performanc)

    compare_indicator_performance['exp_id']=compare_indicator_performance.index
    return compare_indicator_performance

ray_folder = Path("/home/akister/ray_results")
run_name = "fn_2024-04-29_07-33-50"
old_runs = {
    "fn_2024-04-29_07-33-50":None,
    "fn_2024-05-02_07-25-32":None,
    "fn_2024-05-02_07-24-51":None,}

old_runs = {
    "fn_2024-05-06_13-03-30": "simple MT example",
    "fn_2024-05-06_13-29-31":"simple CM ",
    "fn_2024-05-08_10-53-57":"simple sine",
    "fn_2024-05-07_08-03-04":"simple ns_one_dim",
    "fn_2024-05-07_15-55-44":"continues training",
    "fn_2024-05-15_08-34-13":"Current sine",
    "fn_2024-05-15_09-37-06":"CM2",
    "fn_2024-05-24_08-49-58": "MT with freez by optim",
    "fn_2024-05-24_08-50-43":"CM2 with freez by optim",
    "fn_2024-05-24_12-47-50": "two dim with freez by optim, MT lr",
    "fn_2024-05-24_13-18-54": "more training epochs, but everything else like fn_2024-05-24_12-47-50",
    "fn_2024-05-24_13-38-05": "two dim new lr, everything else like fn_2024-05-24_13-18-54",
    "fn_2024-05-24_15-21-41": "two dim MT params, more epochs and more time!!",
    "fn_2024-05-24_16-53-24": "two dim hyper param search (lr sol, test)",
    "fn_2024-05-27_09-13-17": "two dim good setting",
    "fn_2024-05-27_13-47-46": "two dim 10 times weight_soln_obs, 0.01 times lrs",
    "fn_2024-05-27_14-32-32": "two dim 10 times weight_soln_obs, 0.001 times lrs, 0.1 times diagonal_forcing",
    "fn_2024-05-27_14-45-07": "two dim 10 times weight_soln_obs, 0.001 times lrs",
    "fn_2024-05-27_16-08-09": "two dim lr_soln: 0.0009, lr_testfn:  0.00009",
    "fn_2024-05-28_15-58-25": "HP search",
    "fn_2024-05-29_11-12-32": "HP search II, more epochs",
    "fn_2024-05-29_13-05-05": "HP search III, more repetitions, best diff: 0.04",
    "fn_2024-05-29_14-31-58": "ns_two_dim with optimizer freez",
    "fn_2024-06-04_10-57-20": "ns_two_dim HP search",}
start_close_to_obst={
    "fn_2024-06-19_12-02-28":"clos to obs CM2, 4.1.3 1D piecewise example",
    "fn_2024-06-19_12-09-31":"clos to obs sine NOT in OVERLEAVE",
    "fn_2024-06-19_12-15-54":"clos to obs ns_one_dim 4.1.2 1D non-symmetric example",}
overleafe_runs = {
    "fn_2024-06-19_12-45-32":"CM2, 4.1.3 1D piecewise example",
    "fn_2024-06-19_12-52-12":"sine NOT in OVERLEAVE",
    "fn_2024-06-19_12-58-34":"ns_one_dim 4.1.2 1D non-symmetric example",
    #"fn_2024-06-19_09-17-30":"two_dim",
    #"fn_2024-06-19_08-46-00":"MT",
}
continue_runs = {
    "fn_2024-06-19_18-42-17":"CM2, 4.1.3 1D piecewise example",
    "fn_2024-06-19_18-54-07":"sine NOT in OVERLEAVE",
    "fn_2024-06-19_19-05-13":"ns_one_dim 4.1.2 1D non-symmetric example",}
identical_config_runs = {
    "fn_2024-06-20_09-38-39":"CM2, 4.1.3 1D piecewise example",
    "fn_2024-06-20_09-45-33":"one_dim, 4.1.1",
    "fn_2024-06-20_09-51-43":"sine NOT in OVERLEAVE",
    "fn_2024-06-20_09-58-00":"ns_one_dim 4.1.2 1D non-symmetric example",}
individual_config= {
    "fn_2024-06-20_15-22-56": "CM2, 4.1.3 1D piecewise example",
    "fn_2024-06-20_15-32-58":"one_dim, 4.1.1",
    "fn_2024-06-20_16-06-33":"sine NOT in OVERLEAVE",
    "fn_2024-06-20_16-41-07":"ns_one_dim 4.1.2 1D non-symmetric example",}
sigmoid={
    "fn_2024-06-21_09-42-00":"CM2, 4.1.3 1D piecewise example",
    "fn_2024-06-21_09-52-21":"one_dim, 4.1.1",
    "fn_2024-06-21_10-01-47":"sine NOT in OVERLEAVE",
}
runs_dict = identical_config_runs #sigmoid #identical_config_runs #dict(overleafe_runs,**continue_runs) # overleafe_runs # old_runs #



interesting_runs=list(runs_dict.keys())
st.write(runs_dict)

with st.container():
    run_name = st.selectbox(
        'What ray run do you like to analyze?',
        interesting_runs,
        index =0)

    max_len = st.number_input(label='How many experiments should we consider (maximally)?',
                              min_value =1,value = 10, step =1)
    
    results = get_results(ray_folder,run_name,max_len=max_len) 
    st.write(f'Data of run {run_name} \n Example: {results.get_example().name}')

    with st.expander('The config'):
        config_as_pd=pd.DataFrame.from_dict(results.get_cfg_content(), orient='index')
        st.table(config_as_pd)

with st.container(border = True):
    metric_for_hist = st.selectbox(
        'Choos the metric for the histogram',
        eval_r.possible_metrics)
    data_for_hist = results.get_final_errors(metric=metric_for_hist)
    
    fig = ff.create_distplot([[v for v in list(data_for_hist.values()) if v is not None]],
                             [metric_for_hist],bin_size=[.1])

    st.plotly_chart(fig, use_container_width=True)
    ds = pd.DataFrame.from_dict(data_for_hist, orient='index')
    with st.expander('Data for this plot'):
        st.table(ds)

with st.container(border = True):
    st.write('What metric is a good indicator for the performance?')
    
    metric_for_performance = st.selectbox(
        'Metric for Performance',
        eval_r.possible_metrics,
        index = 4)

    metric_for_indicator = st.selectbox(
        'Metric for indicator',
        eval_r.possible_metrics,
        index = 0)

    genre = st.radio(
        "Final or Initial value",
        ["Final", "Initial"],)    
    data_for_ind_per=[]
    for nbr,metric in enumerate([metric_for_indicator,metric_for_performance]):
        if nbr == 0 and genre == "Initial":
            data_in_dict = results.get_initial_errors(metric=metric)
        else: 
            data_in_dict = results.get_final_errors(metric=metric)

        
        data_dict_list=[{'exp_id':k,metric:v} for k,v in  data_in_dict.items()]
        data_in_pd = pd.DataFrame(data_dict_list).set_index('exp_id')
        data_for_ind_per.append(data_in_pd)
    ds_ind_per = data_for_ind_per[0].join(data_for_ind_per[1])
    ds_ind_per['exp_id']=ds_ind_per.index
    
    plot_x_lower_bound = st.number_input(
        'Lower bound for the x axis of  the plot',
        value=ds_ind_per[metric_for_indicator].quantile(q=0.25))
    plot_x_upper_bound = st.number_input(
        'Upper bound for the x axis of  the plot',
        value=ds_ind_per[metric_for_indicator].quantile(q=0.75))
    x_valls_zoom = [plot_x_lower_bound,plot_x_upper_bound]
    idx_zoom = (
        (ds_ind_per[metric_for_indicator]>x_valls_zoom[0])
        *(ds_ind_per[metric_for_indicator]<x_valls_zoom[1]))

    
   
    c = (
        alt.Chart(ds_ind_per[idx_zoom])
        .mark_circle()
        .encode(x=metric_for_indicator, 
                y=metric_for_performance,  
                tooltip=[metric_for_indicator, 
                         metric_for_performance, 
                         'exp_id'])# size="c", color="c",
    )
    st.altair_chart(c, use_container_width=True)



with st.container(border = True):
    if st.button('Delete the cashed matrix?'):
        use_existing_cash = False
    else:
        use_existing_cash = True
    
    loss_matrix_soln_test=get_loss_matrix_soln_test(
        ray_folder,
        run_name,
        max_len,
        use_existing_cash)
    
    c = (
        alt.Chart(loss_matrix_soln_test).mark_rect().encode(
            alt.X("test_id").title("Test"),
            alt.Y("sol_id").title("Soln"),
            alt.Color("loss_r").title("loss_r"),
        ))
    
    st.altair_chart(c, use_container_width=True)

possible_matrix_metric = ['max','min','mean','std']

with st.container(border = True):
    
    st.write('What sol-test matrix metric  is a good indicator for the performance?')
    metric_for_indicator = st.selectbox(
        'Metric to apply to the sol-text matrix (this will be the indicator)',
        possible_matrix_metric,
        index = 0)

    metric_for_performance = st.selectbox(
        'Metric for Performance',
        eval_r.possible_metrics,
        index = 4,
        key='performance_in_matrix_comparison')

    compare_indicator_performance = get_matrix_metric_vs_performanc(
        ray_folder,
        run_name,
        max_len,
        metric_for_indicator,
        metric_for_performance,
        use_existing_cash
    )

    c = (
        alt.Chart(compare_indicator_performance)
        .mark_circle()
        .encode(x='loss_r',
                y=metric_for_performance,  
                tooltip=['loss_r', 
                         metric_for_performance, 
                         'exp_id'])# size="c", color="c",
    )
    st.altair_chart(c, use_container_width=True)
def plot_surface(data,pdim):
    if data is None or pdim <2:
        st.write("No Data")
    else:
        figs = []
        names=[]
        for fn_data in  data:
            if fn_data is None:
                break
            name = fn_data['name'][0]
                
            x = fn_data['X_x']
            y = fn_data['X_y']
            z = fn_data['value']
            names.append(name)
            
            trace = go.Surface(x=x, y=y, z=z, name=name)
            fig = go.Figure(data=trace)
            # Customize the plot layout
            fig.update_layout(
            title=f'{name}',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title=name
                )
            )
                
            figs.append(fig)
            
        cols = st.columns(2)
        for i,fig in enumerate(figs):
            with cols[i%2]:
                st.plotly_chart(fig)
                
                
                
                    

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


    
pdim2nb_grid_points = {1: 1000, 2:1000}
with st.container(border = True):
    results = get_results(ray_folder,run_name,max_len=max_len)
    pdim = results.get_dim()
    experiment_id = st.selectbox(
        'Experiment for plotting',
        results.get_experiments().keys(),
        index = 0)
    experiment = results.get_experiments()[experiment_id]
    with st.container():
        progress_data = experiment.get_training_process()
        columns = st.multiselect('Select columns to plot', progress_data.columns)
    
    
        if len(columns) > 0:
            progress_plt_data=[]
            for c in columns:
                d = (progress_data[["epoch",c]]).rename(columns={c:'value'}).assign(name=c)
                progress_plt_data.append(d)
            progress_plt_df = pd.concat(progress_plt_data)
            
            plot_y_lower_bound = st.number_input('Lower bound for the y axis of  the plot',
                                                 value=progress_plt_df['value'].quantile(q=0.25))
            plot_y_upper_bound = st.number_input('Upper bound for the y axis of  the plot',
                                                 value=progress_plt_df['value'].quantile(q=0.75))
            y_valls_zoom = [plot_y_lower_bound,plot_y_upper_bound]
            
            idx_zoom = (progress_plt_df["value"]>y_valls_zoom[0])*(progress_plt_df["value"]<y_valls_zoom[1])
            line = alt.Chart(progress_plt_df[idx_zoom]).mark_line().encode(
                x='epoch:O',
                y='value:Q',
                color='name:N'
            )
            st.altair_chart(line, use_container_width=True)
        else:
            st.write('Please select at least one column to plot.')

    grid = experiment.get_regular_grid(n_interior=pdim2nb_grid_points[pdim],n_boundary=10)
    example = experiment.get_example()
    with st.container():
        soln_nn=experiment.get_soln()
        if soln_nn is not None:
            gt=example.exact_solution
            fn_s={'sol_nn':soln_nn,
                  'gt_nn':gt,
                  'diff': diff(soln_nn,gt),
                  'test_nn':experiment.get_test(),
                  'obstacle_nn': example.obstacle,}
            data=[]
            meta_data = []
            meta_data_entry = {
                'ray_folder':ray_folder,
                'run_name':run_name,
                'experiment_id':experiment_id,
                'epoch': 'final',
                'fn_name': None,}
            for name,fn in fn_s.items():
                fn_data=eval_r.plot_data_for_nn_on_mesh(fn,grid,pdim=pdim,name=name)
                data.append(fn_data)
                meta_data_entry['fn_name'] = name
                meta_data.append(meta_data_entry.copy())
            if pdim==1:
                df = pd.concat([pd.DataFrame(entry) for entry in data])
                plot_line_data(df,pdim)
                data_for_plot = df.pivot(index='X_x',
                                         columns="name",
                                         values='value')
                lp=pm.LinePlot(ray_folder=ray_folder, 
                               run_name=run_name,
                               data=data_for_plot,
                               experiment_id=experiment_id,
                               epoch=None)
                st.write(lp.generte_tikz_code())
                if st.button("Save the plot of the functions?"):
                    lp.write_plot()
                    st.write("The data for ploting is in the files:")
                    st.write(lp.get_file_adresses())
                    
                    
                else:
                    st.write('You did not save the plot')
                if st.button("Run tectonic?"):
                    output = lp.execute_tikz_standalown()
                    if output is None:
                        st.write('Some thing went wrong wiht the tikz to pdf transformation')
                    else:
                        st.write(output.stdout)
                        st.write(output.stderr)
                        pdf_viewer(str(lp.get_pdf_of_tikz()))
                else:
                    st.write("To make the pdf run: tectonic standalown_figur.tikz; then you can press the button to display the result.")
            elif pdim ==2:
                plot_surface(data,pdim)
                c_plots = pm.CollectionOf2DFunctions(data,meta_data)
                c_plots.write_collection()
                st.write(c_plots.get_plot_adresses())
                c_plots.execute_tikz_standalown()
                for adress in c_plots.get_plot_adresses():
                    pdf_viewer(str(adress))
        else:
            st.write('No final checkpoint')

    
with st.container(border = True):
    results = get_results(ray_folder,run_name,max_len=max_len)
    pdim = results.get_dim()
    
    experiment_id_for_history = st.selectbox(
        'Experiment for plotting the history',
        results.get_experiments().keys(),
        index = 0)
    experiment_for_history = results.get_experiments()[experiment_id_for_history]

    history = experiment_for_history.get_check_hist()
    historic_chpts = st.multiselect('Select chpt epochs to plot', [f['epoch'] for f in history])
    grid = experiment_for_history.get_regular_grid(n_interior=pdim2nb_grid_points[pdim],n_boundary=10)
    example = experiment_for_history.get_example()
    if len(historic_chpts)>0:
        filtered_hist = [f for f in history if f['epoch']  in historic_chpts]
    else:
        filtered_hist = [history[0]]
    for h_chpt in filtered_hist:
        epoch = h_chpt['epoch']
        st.write(f'{epoch}')
        with st.container():
            soln_nn=h_chpt['chpt'].get_soln()
            gt=example.exact_solution
            fn_s={'sol_nn':soln_nn,
                  'gt_nn':gt,
                  'diff': diff(soln_nn,gt),
                  'test_nn':h_chpt['chpt'].get_test(),
                  'obstacle_nn': example.obstacle,}
            data=[]
            for name,fn in fn_s.items():
                fn_data=eval_r.plot_data_for_nn_on_mesh(fn,grid,pdim=pdim,name=name)
                data.append(fn_data)
            
            if pdim == 1:
                df = pd.concat([pd.DataFrame(entry) for entry in data])
                plot_line_data(df,pdim)
            elif pdim == 2:
                plot_surface(data,pdim)


    
