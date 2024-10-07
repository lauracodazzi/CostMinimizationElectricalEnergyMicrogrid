# -*- coding: utf-8 -*-

import os.path 
from auxiliary_functions import *
from multiprocessing import Pool
import matplotlib.pyplot as plt



f = open("data/load", "r") 
load = f.readlines()
f.close()

f = open("data/forecast", "r") 
forecast = f.readlines()
f.close()

# number of prosumers
N = len(load)

# lenght of the time horizon
T = len(convert2float(load[0], separator = "\t"))

all_instances = []


for n in range(N):
    l = convert2float(load[n].replace("\n", "\t"), separator = "\t")[:(T)]
    e = convert2float(forecast[n].replace("\n", "\t"), separator = "\t")[:(T)]
    i = Instance()
    i.T = T
    i.demand = l
    i.energy = e
    i.initial_battery = 0
    i.capacity = 8
    all_instances = np.append(all_instances, i)

total_daily_demand = np.zeros(T)
total_daily_supply = np.zeros(T)

for t in range(T):
    for n in range(N):
        total_daily_demand[t] = total_daily_demand[t] + all_instances[n].demand[t]
        total_daily_supply[t] = total_daily_supply[t] + all_instances[n].energy[t]

excess_energy = total_daily_demand-total_daily_supply
ratio_energy = total_daily_supply/total_daily_demand

excess_energy = total_daily_demand-total_daily_supply
ratio_energy = total_daily_supply/total_daily_demand

max_excess = max(ratio_energy)
min_excess = min(ratio_energy)

prices = [32, 31.72, 32.15, 32.7, 33.29, 34.92, 36.61, 36.35, 34.36, 30.74,\
          27.04, 23.40, 22.57, 22.43, 22.57, 22.43, 22.43, 24.06, 27.42, \
          32.01, 35.11, 38.30, 44.15, 40.54, 36.53, 34.58]

r = np.mean(prices)

# here I linearize the epsilon in function of the excess
m = r/(min_excess-max_excess)
q = -max_excess*m

eps = q*np.ones(T) + m*ratio_energy

for n in range(N):
    all_instances[n].p_buy = r*np.ones(T)
    all_instances[n].p_sell = r*np.ones(T) - eps

result = []

# if __name__ == "__main__":
#     with Pool() as pool:
#       result = list(pool.map(P2, all_instances))
#       # LP_partial = partial(LP, P = 1)
#       # result_LP = pool.map(LP_partial, all_instances)
#     print("Program finished!")


for idx in range(N):
    result.append(P2(all_instances[idx]))

idx = 0
a1= all_instances[idx].demand
b1 = all_instances[idx].energy 
 
r1 = np.arange(len(a1))
width1 = 0.3
 
plt.bar(r1, a1, width=width1)
plt.bar(r1 + width1, b1, width=width1)

fig, axs = plt.subplots(3,7, sharex = True, sharey = 'row')    

bars = np.arange(len(all_instances[0].demand))


fig, axs = plt.subplots(3,7, sharex = True, sharey = 'row')    

bars = np.arange(len(all_instances[0].demand))

    
for idx in range(7):
    axs[0][idx].bar(bars, all_instances[idx].demand, width = 0.4)
    axs[0][idx].bar(bars, all_instances[idx].energy, width = 0.4)
    # axs[0][idx-7].bar(len(all_instances[idx].energy), all_instances[idx].energy,width = 0.001)
    axs[1][idx].bar(bars, result[idx][1], width = 0.4)
    axs[1][idx].bar(bars, result[idx][2], width = 0.4)
    axs[2][idx].plot(result[idx][3][list(range(0, len(result[idx][3]), 2))])

fig, axs = plt.subplots(3,7, sharex = True, sharey = 'row')    


for idx in range(7, 14):
    XXX = result[idx]
    axs[1][idx-7].bar(bars, all_instances[idx].demand, width = 0.3)
    axs[1][idx-7].bar(bars+0.2, all_instances[idx].energy, width = 0.4)
    axs[0][idx-7].bar(bars, XXX[1], width = 0.4)
    axs[0][idx-7].bar(bars+0.2, XXX[2], width = 0.4)
    axs[2][idx-7].plot(XXX[3][list(range(0, len(XXX[3]), 2))])
