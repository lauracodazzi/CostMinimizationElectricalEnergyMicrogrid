# -*- coding: utf-8 -*-

import os.path 
from auxiliary_functions import *
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np

#Set here the desired capacity for the prosumers
capacity = 8

# Import of the data and allocate the instances
f = open("data/load", "r") 
load = f.readlines()
f.close()
f = open("data/forecast", "r") 
forecast = f.readlines()
f.close()
f = open("data/prices", "r") 
temp = f.readlines()
prices = convert2float(temp[0], separator = "\t")
f.close()

N = len(load) # number of prosumers
T = len(convert2float(load[0], separator = "\t")) # lenght of the time horizon
all_instances = []

for n in range(N):
    l = convert2float(load[n].replace("\n", "\t"), separator = "\t")[:(T)]
    e = convert2float(forecast[n].replace("\n", "\t"), separator = "\t")[:(T)]
    i = Instance()
    i.T = T
    i.demand = l
    i.energy = e
    i.initial_battery = 0
    i.capacity = capacity
    all_instances = np.append(all_instances, i)

total_daily_demand = np.zeros(T)
total_daily_supply = np.zeros(T)

for t in range(T):
    for n in range(N):
        total_daily_demand[t] = total_daily_demand[t] + all_instances[n].demand[t]
        total_daily_supply[t] = total_daily_supply[t] + all_instances[n].energy[t]

# Fit of the prices
excess_energy = total_daily_demand-total_daily_supply
ratio_energy = total_daily_supply/total_daily_demand
max_excess = max(ratio_energy)
min_excess = min(ratio_energy)
r = np.mean(prices)
m = r/(min_excess-max_excess)
q = -max_excess*m
eps = q*np.ones(T) + m*ratio_energy
result = []

for n in range(N):
    all_instances[n].p_buy = r*np.ones(T)
    all_instances[n].p_sell = r*np.ones(T) - eps
    result.append(P2(all_instances[n]))

# Prices plot
figure = plt.figure()
a1= total_daily_demand
b1 = total_daily_supply
r1 = np.arange(len(a1))
plt.bar(r1, b1, width=0.3, label="supply")
plt.bar(r1+0.3, a1, width=0.3, label="demand")
plt.plot(all_instances[0].p_sell, label="prices")
plt.legend()
plt.xlabel("Hours")
plt.savefig("Plots/prices.png",  bbox_inches='tight')
plt.savefig("Plots/prices.svg",  bbox_inches='tight')
plt.show()

# Prosumers profile plots
fig, axs = plt.subplots(3,7, sharex = True, sharey = 'row')    
bars = np.arange(len(all_instances[0].demand)) 
for idx in range(7):
    axs[0][idx].bar(bars, all_instances[idx].demand, width = 0.4)
    axs[0][idx].bar(bars, all_instances[idx].energy, width = 0.4)
    # axs[0][idx-7].bar(len(all_instances[idx].energy), all_instances[idx].energy,width = 0.001)
    axs[1][idx].bar(bars, result[idx][1], width = 0.4)
    axs[1][idx].bar(bars, result[idx][2], width = 0.4)
    axs[2][idx].plot(result[idx][3][list(range(0, len(result[idx][3]), 2))])
fig.text(0.5, 0.04, 'Time period', ha='center', va='center')
fig.text(0.06, 0.75, 'Energy\nProfile', ha='center', va='center', rotation='vertical')
fig.text(0.06, 0.5, 'Bought\nSold', ha='center', va='center', rotation='vertical')
fig.text(0.06, 0.25, 'Battery', ha='center', va='center', rotation='vertical')
fig.savefig("Plots/all_prosumers1_battery" + str(all_instances[0].capacity) + ".svg",  bbox_inches='tight')
fig.savefig("Plots/all_prosumers1_battery" + str(all_instances[0].capacity) + ".png",  bbox_inches='tight')

fig, axs = plt.subplots(3,7, sharex = True, sharey = 'row')    
for idx in range(7, 14):
    XXX = result[idx]
    axs[1][idx-7].bar(bars, all_instances[idx].demand, width = 0.3)
    axs[1][idx-7].bar(bars+0.2, all_instances[idx].energy, width = 0.4)
    axs[0][idx-7].bar(bars, XXX[1], width = 0.4)
    axs[0][idx-7].bar(bars+0.2, XXX[2], width = 0.4)
    axs[2][idx-7].plot(XXX[3][list(range(0, len(XXX[3]), 2))])
fig.text(0.5, 0.04, 'Time period', ha='center', va='center')
fig.text(0.06, 0.75, 'Energy\nProfile', ha='center', va='center', rotation='vertical')
fig.text(0.06, 0.5, 'Bought\nSold', ha='center', va='center', rotation='vertical')
fig.text(0.06, 0.25, 'Battery', ha='center', va='center', rotation='vertical')
fig.savefig("Plots/all_prosumers2_battery" + str(all_instances[0].capacity) + ".svg",  bbox_inches='tight')
fig.savefig("Plots/all_prosumers2_battery" + str(all_instances[0].capacity) + ".png",  bbox_inches='tight')