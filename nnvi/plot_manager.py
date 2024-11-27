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
from nnvi.eval_ray_run import ray_files_adresses

import nnvi as root_module

from pathlib import Path
import json
import pandas as pd
import numpy as np

import pickle
import subprocess

ray_folder = Path("/home/akister/ray_results")
run_name = "fn_2024-04-25_15-38-17"#"fn_2024-04-24_15-03-26"#"fn_2024-04-15_09-27-29"


class PlotData():
    def __init__(self, ray_folder, run_name,data,experiment_id=None,epoch=None):
        self.ray_folder = ray_folder
        self.run_name = run_name
        self.plot_name = None
        self.file_dict = self.get_ray_folder_adress()
        self.plot_folder = self.get_plot_folder()
        self.data = data
        self.experiment_id = None
    
    def get_ray_folder_adress(self):
        return Path(self.ray_folder)/self.run_name

    def get_plot_folder(self):
        if self.plot_name == None:
            self.plot_folder = None
        else:
            self.plot_folder = Path(self.get_ray_folder_adress())/'tikz_plots'
            if self.experiment_id is not None:
                if epoch is None:
                    self.plot_folder=self.plot_folder/self.experiment_id/self.plot_name
                else:
                    self.plot_folder=self.plot_folder/self.experiment_id/self.plot_name/str(epoch)
            else:
                self.plot_folder=self.plot_folder/self.plot_name
        if self.plot_folder is not None:
            self.plot_folder.mkdir(parents=True, exist_ok=True)
        return self.plot_folder 

    def get_data_adress(self):
        self.plot_folder = self.get_plot_folder()
        return self.plot_folder / "data.csv"
    def get_tikz_code(self):
        self.plot_folder = self.get_plot_folder()
        return self.plot_folder/ "figur.tikz"
    def get_stand_alown_tikz_code(self):
        self.plot_folder = self.get_plot_folder()
        return self.plot_folder/ "standalown_figur.tikz"

    def get_pdf_of_tikz(self):
        self.plot_folder = self.get_plot_folder()
        return self.plot_folder/"standalown_figur.pdf"
        
    def data_has_correct_form(self):
        correct = False
        correct_typ = isinstance(self.data, pd.DataFrame)
        correct_columns = False
        return (correct_typ and correct_columns)

    def write_data(self):
        if not self.data_has_correct_form():
            print('Data has the wrong form')
        else:
            self.data.to_csv(self.get_data_adress(), 
                             sep=',', 
                             index=True,
                             header=True)
        
    def generte_tikz_code(self):
        return None
    def generate_standalown_tikz_code(self):
        tikz_code = self.tikz_in_outro["standalownIn"] 
        tikz_code= tikz_code + self.generte_tikz_code()
        return tikz_code+self.tikz_in_outro["standalownOut"]
        
    def write_tikz_code(self):
        tikz_code = self.generte_tikz_code()
        with open(self.get_tikz_code(), 'w') as tikzfile:
            tikzfile.write(tikz_code)
    def write_standalown_tikz_code(self):
        tikz_code = self.generate_standalown_tikz_code()
        with open(self.get_stand_alown_tikz_code(), 'w') as tikzfile:
            tikzfile.write(tikz_code)
    def write_plot(self):
        self.write_data()
        self.write_tikz_code()
        self.write_standalown_tikz_code()
    def get_file_adresses(self):
        return [self.get_data_adress(),self.get_tikz_code(),self.get_stand_alown_tikz_code()]
    def execute_tikz_standalown(self):
        if self.get_stand_alown_tikz_code().exists():
            adress = self.get_stand_alown_tikz_code()
            output = subprocess.run(["tectonic",str(adress)], capture_output=True, text=True)
        else:
            output = None
        return output

class LinePlot(PlotData):
    def __init__(self,ray_folder, 
                 run_name,
                 data,
                 experiment_id=None,
                 epoch=None):
        super().__init__(ray_folder, run_name,data,experiment_id = experiment_id)        
        self.plot_name = 'sol_test_gt_'
        if epoch==None:
            self.plot_name = self.plot_name +'final'
        else:
            self.plot_name = self.plot_name +str(epoch)
        self.plot_properties={
            'sol_nn':{'color':'red'},
            'gt_nn':{'color':'magenta'},
            'diff': {'color':'blue'},
            'test_nn':{'color':'brown'},
            'obstacle_nn': {'color':'black'}}
        self.tikz_in_outro = {
            'standalownIn':r"""
\documentclass{standalone}
\usepackage{tikz}
\usepackage{pgfplots}

\begin{document}
"""
            ,
            'intro':r"""
\begin{tikzpicture}
\begin{axis}[
xlabel=$x$,
ylabel=$y$,
legend pos=north west,
]
            """
            ,
            'outro':r"""
\end{axis}
\end{tikzpicture}
            """
            ,
            'standalownOut':r"""
\end{document}
            """}
    def data_has_correct_form(self):
        correct = False
        correct_typ = isinstance(self.data, pd.DataFrame)
        correct_columns = True
        return (correct_typ and correct_columns)

    def tikz_line(self,x_name,col_name):
        data_file=self.get_data_adress()
        color = self.plot_properties[col_name]['color']
        return f"\\addplot[mark=none,color={color}] table [x={x_name}, y={col_name}, col sep=comma, header=has colnames] {{\"{data_file}\"}}; \n"
    def generte_tikz_code(self):
        tikz_code = self.tikz_in_outro["intro"]
        x_name=self.data.index.name
        for col_name in list(self.data.columns):
            tikz_code = tikz_code + self.tikz_line(x_name,col_name)
        return tikz_code + self.tikz_in_outro["outro"]

    

class SurfacePlot(PlotData):
    def __init__(self,ray_folder, 
                 run_name,
                 data,
                 experiment_id=None,
                 epoch=None,
                 fn_name=None):
        super().__init__(ray_folder, run_name,data,experiment_id = experiment_id)        
        self.plot_name = 'sol_test_gt_'
        if epoch==None:
            self.plot_name = self.plot_name +'final'
        else:
            self.plot_name = self.plot_name +str(epoch)
        self.plot_name = Path(self.plot_name)/fn_name
        self.plot_properties={
            'sol_nn':{'color':'red'},
            'gt_nn':{'color':'magenta'},
            'diff': {'color':'blue'},
            'test_nn':{'color':'brown'},
            'obstacle_nn': {'color':'red'}}
        self.tikz_in_outro = {
            'standalownIn':r"""
\documentclass{standalone}
\usepackage{tikz}
\usepackage{pgfplots}

\begin{document}
"""
            ,
            'intro':r"""
\begin{tikzpicture}
  \begin{axis}[
    width=0.8\textwidth,
    height=0.6\textwidth,
    xlabel=$x$,
    ylabel=$y$,
    zlabel=$z$,
    view={60}{30},
    colormap/jet,
  ]
            """
            ,
            'outro':r"""
\end{axis}
\end{tikzpicture}
            """
            ,
            'standalownOut':r"""
\end{document}
            """}
    def data_has_correct_form(self):
        correct = False
        correct_typ = isinstance(self.data, pd.DataFrame)
        correct_columns = True
        return (correct_typ and correct_columns)

    def tikz_line(self):
        data_file=self.get_data_adress()
        return f"\\addplot3[surf,shader=mean , faceted color=blue] table[x=X_x, y=X_y, z=value, col sep=comma, header=has colnames] {{\"{data_file}\"}}; \n"
    def generte_tikz_code(self):
        tikz_code = self.tikz_in_outro["intro"]
        tikz_code = tikz_code + self.tikz_line()
        return tikz_code + self.tikz_in_outro["outro"]
            

class CollectionOf2DFunctions():
    def __init__(self,data,meta_data):
        self.data=data # list of 2-dim data sets
        self.meta_data=meta_data # list of meta data dicts, the order should be inline with self.data
        self.plots=[]
        for d,m in zip(self.data,self.meta_data):
            d_transformed = {'X_x':d['X_x'].flatten(),
                             'X_y':d['X_y'].flatten(),
                             'value':d['value'].flatten(),}
            self.plots.append(SurfacePlot(
                ray_folder=m['ray_folder'], 
                run_name=m['run_name'],
                data=pd.DataFrame(d_transformed),
                experiment_id=m['experiment_id'],
                epoch=m['epoch'],
                fn_name=m['fn_name']))
        
    def write_collection(self):
        [p.write_plot() for p in self.plots]
    def execute_tikz_standalown(self):
        [p.execute_tikz_standalown() for p in self.plots]
    def get_plot_adresses(self):
        return [p.get_pdf_of_tikz() for p in self.plots]
