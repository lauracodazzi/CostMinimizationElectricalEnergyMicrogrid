# -*- coding: utf-8 -*-
"""
Generating and launching instances
@author: codaz
"""

import numpy as np
import matplotlib.pyplot as plt
from auxiliary_functions import run_comparison


# Legend of the algorithms:
#   0:  PEACS-B-U
#   1:  PEACS-BS-U
#   2:  PEACS-B
#   3:  PEACS-BS
#   4:  PEACS-B-Ex
#   5:  PEACS-BS-Ex

gurobi_time_limit = False
plot_result = True
Folder_name = "Results_file/"
T_min = 100
T_max = 10000
skip = 195
algs = [0,1,2,3]
capacity = [1000]
seeds = [10958, 58964, 63895, 85245, 95622, 48893, 28054, 88592, 80502, 89495]
extra_seeds = 10



run_comparison(gurobi_time_limit = gurobi_time_limit, plot_result=plot_result,\
               Folder_name=Folder_name, T_min=T_min, T_max=T_max,\
               skip=skip, algs=algs, capacity=capacity, seeds=seeds,\
               extra_seeds=extra_seeds)

          

