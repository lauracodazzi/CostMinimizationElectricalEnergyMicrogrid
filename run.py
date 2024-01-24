# -*- coding: utf-8 -*-
"""

Code for runing the sperimental analysis of section 7 of the paper 
"Efficient cost-minimization schemes for electrical energy demand
satisfaction by prosumers in microgrids with battery storage
capabilities"

"""

from auxiliary_functions import run_comparison

# Time limit for the resolution of the ILP instance in seconds.
# If set to 'False', no time limits will be applied
gurobi_time_limit = False 

# plot_result: boolean to suppress the plots of the final comparison
# If set to 'True' the plots will be displayed and saved automatically in the 
# Plots/folder
plot_result = True

# Parameter to indicate the path for saving the results of the comparison
Folder_name = "Results_file/"

# Minimum time horizon for the instances
T_min = 100

# Maximum time horizon for the instances
T_max = 10000

# Parameter to decrease the total number of generated instances.
# After the instance with time horizon T_min, the instance with time horizon
# T_min + skip will be generated and so on.
skip = 195

# Parameters to select the desired algorithm to compare with the ILP.
# Legend of the algorithms:
#   0:  PEACS-B-U
#   1:  PEACS-BS-U
#   2:  PEACS-B
#   3:  PEACS-BS
#   4:  PEACS-B-Ex
#   5:  PEACS-BS-Ex
algs = [0,1,2,3]

# Parameter to set the upper bound of the simulated capacity. Instances will
# be generated from a uniform distribution with support [0, capacity]
capacity = [1000]

# Random seeds to generate the instances
seeds = [10958, 58964, 63895, 85245, 95622, 48893, 28054, 88592, 80502, 89495]

# Additional random seeds. The total number of generated instances is given
# by the length of the vector seed times the value of the parameter extra_seed
extra_seeds = 10


# Once the above paraemters are set, please run the following function.
run_comparison(gurobi_time_limit = gurobi_time_limit, plot_result=plot_result,\
               Folder_name=Folder_name, T_min=T_min, T_max=T_max,\
               skip=skip, algs=algs, capacity=capacity, seeds=seeds,\
               extra_seeds=extra_seeds)
    
    