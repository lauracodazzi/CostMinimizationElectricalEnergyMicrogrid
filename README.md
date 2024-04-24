# EfficientCostminiMizazionforElettricalEnergyDemand
Here the source for replicating the Experimental Analysis of the paper "Efficient cost-minimization schemes for electrical energy demand  satisfaction by prosumers in microgrids with battery storage capabilities".
In the file *algorithms.py*, the 6 combinatorial algorithms are implemented.
By executing the code, it is possible to compare each one of the six algorithms with the ILP solved with Gurobi.
Please, make sure to create locally two folders to store the results:
- the folder 'Plots/' to store a graphical representation of the analysis
- the folder 'Results_file/' to store the results of the comparison. This can be however be customized in the file *run.py*
To execute the code it is necessary to set the desired parameters in the file *run.py* and run the file.
The explanation of the parameters is provided in the *run.py* file.
To ensure the proper execution of the code, make sure that the solver Gurobi is installed.
