import os
import matplotlib as mpl

path_save = 'Simulations/'
path_graph = 'Figures/'

# Create directories if they do not exist
os.makedirs(path_save, exist_ok=True)
os.makedirs(path_graph, exist_ok=True)

    
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

defaultcolors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


