# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
import time
import os.path 
# import random as rn
import scipy.stats as stats
import math

from algorithms import IP, P1, P2, P3, P4, P5, P6

def compare_alg(alg, i, time_limit):
    # our_st_p = time.process_time()
    # [cost, x, y, SOC, R, feas] = ([P1,P2,P3,P4_fast,P5,P6][alg])(i)
    # our_et_p = time.process_time()
    if alg == 0:
        our_st_p = time.process_time()
        [cost, x, y, SOC, R, feas] = P1(i) 
        our_et_p = time.process_time()
    if alg == 1:
        our_st_p = time.process_time()
        [cost, x, y, SOC, R, feas] = P2(i) 
        our_et_p = time.process_time()
    if alg == 2:
        our_st_p = time.process_time()
        [cost, x, y, SOC, R, feas] = P3(i) 
        our_et_p = time.process_time()
    if alg == 3:
        our_st_p = time.process_time()
        [cost, x, y, SOC, R, feas] = P4(i) 
        our_et_p = time.process_time()
    if alg == 4:
        our_st_p = time.process_time()
        [cost, x, y, SOC, R, feas] = P5(i) 
        our_et_p = time.process_time()
    if alg == 5:
        our_st_p = time.process_time()
        [cost, x, y, SOC, R, feas] = P6(i) 
        our_et_p = time.process_time()
    our_time_p = our_et_p-our_st_p
    st_p = time.process_time()
    [cost_IP, x_ip, y_ip, SOC_ip, D_soc_ip, feas_IP] = IP(i, P = alg, time_limit = time_limit)
    et_p = time.process_time()
    IP_time_p = et_p-st_p    
    comparison = 1
    if feas_IP == 2:
        IP_time_p = time_limit
    else:
        if abs(feas-feas_IP)>0.5:
            comparison = 0
            print("Infeasibility mismatch with the Instance with seed "+str(i.seed)+" and time horizon " + str(i.T))
        if abs(cost) > 0.0001 and abs(100*(cost-cost_IP)/cost)>0.1 and feas_IP == 1 :
            comparison = 0
            print("Non optimality for the Instance with seed "+str(i.seed)+" and time horizon " + str(i.T))
    partial_name = "N"+str(i.T)+"_C"+str(i.capacity)+"_seed"+str(i.seed) 
    line = partial_name + "\t" + str(comparison) + "\t" + str(feas_IP) + "\t" + str(feas) + "\t" + str(cost_IP) + "\t" + str(cost) + "\t" + str(IP_time_p) + "\t" + str(our_time_p)
    return line, IP_time_p, our_time_p, cost_IP, x_ip, y_ip, SOC_ip, D_soc_ip, feas_IP, cost, x, y, SOC, R, feas



def mean_from_file(file_name, gurobi_time_limit = False, plot_result = True, \
                   Folder_name = "Results_file/", T_min = 100, T_max = 625, \
                   skip = 25, algorithms = [1], seeds = [10958, 58964, 63895, \
                   85245, 95622, 48893, 28054, 88592, 80502, 89495], \
                   extra_seeds = 10):
    if extra_seeds == 0:
        print("Invalid value for extra_seed parameter.\nMinimum value possible is 1")
    f = open(file_name, "r") 
    l = f.readlines()
    f.close()
    n_time = (T_max-T_min)//skip
    Times_IP = np.zeros(n_time)
    Times_our = np.zeros(n_time)
    Times_plot = np.zeros(n_time)
    first_line = 4
    N_seeds = len(seeds)
    for n_it in range(n_time):
        time_h = T_min + (n_time - n_it - 1)*skip
        Times_plot[n_time - n_it - 1] = int(time_h)
        avg_IP = 0
        avg_our = 0
        last_line = 4 + (n_it+1)*extra_seeds*N_seeds
        for n in range(first_line, last_line):
            line = l[n]
            line = line.split("\t")
            IP_time = convert2float(line[len(line)-2])
            avg_IP = avg_IP + IP_time
            our_time = convert2float(line[len(line)-1])
            avg_our = avg_our + our_time
        Times_IP[n_time - n_it -1] = avg_IP/(extra_seeds*N_seeds)
        Times_our[n_time - n_it -1] = avg_our/(extra_seeds*N_seeds)
        first_line = last_line + 1
    return Times_IP, Times_our, Times_plot


# def run_comparison(gurobi_time_limit, plot_result, \
#                    Folder_name, T_min, T_max, \
#                    skip, algs, capacity, \
#                    seeds, extra_seeds):
def run_comparison(gurobi_time_limit = 60*3, plot_result = True, \
                    Folder_name = "Results_file/", T_min = 100, T_max = 10000, \
                    skip = 25, algs = [0,1,2,3,4,5], capacity = [1000], \
                    seeds = [10958, 58964, 63895, 85245, 95622, 48893, 28054,\
                    88592, 80502, 89495], extra_seeds = 10):
    for alg in algs:
        for Cap in capacity:
            begin = time.process_time()
            print("Running algorithm P" + str(alg+1))
            N_seeds = len(seeds)
            n_time = (T_max-T_min)//skip
            Times_IP = np.zeros(n_time)
            Times_our = np.zeros(n_time)
            name_results_file = "P"+str(alg+1)+"_MaxCapacity" + str(Cap) + "_Tmin_" + str(T_min) + "_Tmax" + str(T_max)
            
            counter = 0
            while os.path.exists(Folder_name+"/" +name_results_file):
                counter = counter + 1
                name_results_file = name_results_file + "(" + str(counter) + ")" 
            f = open(Folder_name + "/" +name_results_file, "a")
            f.write(str(T_min)+"\n")
            f.write(str(T_max)+"\n")
            f.write(str(seeds)+"\n")
            f.write("Instance \t correct \t feas_IP \t feas_our \t costsIP \t cost_our \t time_IP \t time_our \n")
            f.close()
            infeasible = np.zeros((T_max-T_min)*len(seeds))
            root_name = "Instances/I_"
            save_Instances = False # to avoid saving the file of the instance
            Times_plot = np.zeros(n_time)
            total_seeds = []
            
            for n_it in tqdm(range(n_time)):
                n = T_min + (n_time - n_it - 1)*skip
                Times_plot[n_time - n_it - 1] = int(n)
                avg_IP = 0
                avg_our = 0
                AA = []
                CC = []
                for ss in range(N_seeds):
                    for more_instances in range(extra_seeds):
                        s = seeds[ss] + (int(n**0.5))*more_instances - more_instances
                        total_seeds = np.append(total_seeds, s)
                        AA.append(s)
                        np.random.seed(s)
                        C = np.ceil(np.random.uniform(0, Cap))
                        CC.append(C)
                        np.random.seed(s)
                        soc0 = np.ceil(np.random.uniform(0, C))
                        i = create_input_file(T = n, capacity = C, seed = s, s0=soc0, save = save_Instances)
                        [line, IP_time_p, our_time_p, cost_IP, x_ip, y_ip, SOC_ip, D_soc_ip, feas_IP, cost, x, y, SOC, R, feas] = compare_alg(alg, i, gurobi_time_limit)
                        avg_IP = avg_IP + IP_time_p
                        avg_our = avg_our + our_time_p
                        f = open(Folder_name + "/" + name_results_file, "a")
                        f.write(line+"\n")
                        f.close()
                Times_IP[n_time - n_it - 1] = avg_IP/(N_seeds*extra_seeds)
                Times_our[n_time - n_it - 1] = avg_our/(N_seeds*extra_seeds)
                
            if plot_result:
                plt.figure()
                plt.plot(range(n_time), Times_IP, label="ILP")
                plt.plot(range(n_time), Times_our, label="Our algorithm")
                plt.legend(loc='upper left')
                plt.xticks([0,n_time//4,n_time//2,3*n_time//4, n_time-1], labels=(int(Times_plot[0]/1000),int(Times_plot[n_time//4]/1000),int(Times_plot[n_time//2]/1000),int(3*Times_plot[n_time//4]/1000), int(Times_plot[n_time-1]/1000)), rotation = 'vertical')
                plt.xlabel("Length of time horizon n [10^3]")
                plt.ylabel("Runtime [s]")
                plt.savefig("Plots/" + str(name_results_file) + "_" + str(int(time.time())) , bbox_inches='tight', save = False)
                
                plt.figure()
                plt.plot(range(n_time), Times_IP, '.', label="ILP")
                plt.plot(range(n_time), Times_our, '.', label="Our algorithm")
                plt.legend(loc='upper left')
                plt.xticks([0,n_time//4,n_time//2,3*n_time//4, n_time-1], labels=(int((Times_plot[0])),int(Times_plot[n_time//4]),int(Times_plot[n_time//2]),int(3*Times_plot[n_time//4]), int(Times_plot[n_time-1])))
                plt.xlabel("Length of time horizon n")
                plt.ylabel("Runtime [s]")
                # plt.ylim([-5,20])
                plt.savefig("Plots/stars_" + str(name_results_file) + "_" + str(int(time.time())), bbox_inches='tight', save=False)
            end = time.process_time()
            print("Total time: " + str(end-begin))

    # Here the class "Instance" is defined to keep track of the parameters in the
    # prosumer energy problem. They are:
       
        # Time horizont --> T
        # Length of the single time period --> delta
        # Range prices for buying energy --> p_buy_min, p_buy_max
        # Distribution for buying prices --> dist_buying
        # Range prices for selling energy --> p_sell_min, p_sell_max
        # Distribution for selling prices --> dist_sellling
        # Energy --> energy
        # Demand --> demand
        # Battery capacity --> capacity
        
class Instance:
    def __init__(self):
        self.T = 0
    def input_param(self, T, delta, p_buy_min, p_buy_max, dist_buying, p_sell_min, \
                p_sell_max, dist_selling, energy, demand, capacity, \
                seed = "none", initial_battery = 0):
        self.T = T
        self.delta = delta
        if seed != "none":
            np.random.seed(seed)
        self.p_buy = np.random.uniform(p_buy_min, p_buy_max, size=T)
        self.dist_buying = dist_buying
        self.p_sell = np.random.uniform(p_sell_min, p_sell_max, size=T)
        self.dist_selling = dist_selling
        self.energy = energy
        self.demand = demand
        self.capacity = capacity
        self.p_sell_min = p_sell_min
        self.p_sell_max = p_sell_max
        self.p_buy_min = p_buy_min
        self.p_buy_max = p_buy_max
        self.seed = seed
        self.file_name = ""
        self.initial_battery = initial_battery
    def read_from_file(self, file_name):
        f = open(file_name, 'r')
        contents = f.readlines()
        self.T = convert2int(contents[0])
        self.delta = convert2float(contents[1])
        self.p_buy_min = convert2float(contents[2])
        self.p_buy_max = convert2float(contents[3])
        self.dist_buying = contents[4]
        self.p_sell_min = convert2float(contents[5])
        self.p_sell_max = convert2float(contents[6])
        self.dist_selling = contents[7]
        self.p_buy = np.random.uniform(self.p_buy_min, self.p_buy_max, size=self.T)
        self.p_sell = np.random.uniform(self.p_sell_min, self.p_sell_max, size=self.T)
        self.energy = convert2float(contents[8])
        self.demand = convert2float(contents[9])
        self.capacity = convert2float(contents[10])
        self.seed = convert2int(contents[11])
        if self.seed != "none":
            np.random.seed(self.seed)
        if len(contents) >= 13:
            self.initial_battery = convert2float(contents[12])
            self.p_buy = convert2float(contents[13])
            self.p_sell = convert2float(contents[14])
        f.close()    
        self.file_name = file_name
    def set_energy_demand(self, d, e):
        self.demand = d
        self.energy = e
    def print_instance(self):
        self.print_prices()
        self.print_energy()
    def print_prices(self):
        plt.figure()
        plt.plot(range(self.T), self.p_buy, label="Price for buying")
        plt.plot(range(self.T), self.p_sell, label="Price for selling")
        plt.ylim(0, np.ceil(self.p_buy_max)+0.7)
        plt.legend(loc='upper right')
        plt.xlabel("Days")
        plt.ylabel("Price for Energy [kW]")
        plt.savefig(self.file_name+'_prices.png')
    def print_energy(self):
        plt.figure()
        plt.plot(self.energy, label="Energy from PV")
        plt.plot(self.demand, label="Demand")
        plt.legend(loc='upper right')
        plt.xlabel("Days")
        plt.ylabel("Scaled Energy [kW]")
        plt.ylim(np.min(self.energy)-0.2,np.max(self.energy)+0.7)
        plt.savefig(self.file_name+'_energydemand.png')

def create_input_file(T = 20, delta = 1, capacity = 10, seed = 1809, s0 = 0,\
    other_attributes = "", p_buy_min = 13, p_buy_max = 20, p_sell_min = 6, \
    p_sell_max = 12, min_diff = -10, max_diff = 10, save = True):

    # Temporary horizon
    # lenght of each time period
    # Capacity of the battery
    # seed 
    # Initial battery

    # Here I simulate the delta energy
    np.random.seed(seed)
    min_diff = -capacity/6
    max_diff = capacity/3
    diff = np.random.uniform(min_diff, max_diff, size=T)
    demand = np.empty(T)

    # Energy produced at time t
    np.random.seed(seed)
    mu_energy = capacity/4
    sigma_energy = capacity**0.5
    energy = np.random.normal(mu_energy, scale=sigma_energy, size=T)
    for idx in range(T):
        energy[idx] = int(energy[idx])
        demand[idx] = energy[idx] + int(diff[idx])
        
    # x_en = np.linspace(mu_energy - 3*sigma_energy, mu_energy + 3*sigma_energy, 100)
    # plt.figure()
    # plt.plot(x_en, stats.norm.pdf(x_en, mu_energy, sigma_energy))

    # Prices for buying energy
    mu_buy = (p_buy_min+p_buy_max)/2
    variance_buy = p_buy_min
    sigma_buy = math.sqrt(variance_buy)
    np.random.seed(seed)    
    p_buy = np.random.normal(mu_buy, sigma_buy, size=T)
    
   
    

    # Prices for selling energy
    mu_sell = (p_sell_min+p_sell_max)/2
    variance_sell = (p_sell_min+p_sell_max)/3
    sigma_sell = math.sqrt(variance_sell)
    np.random.seed(seed)
    p_sell = np.random.normal(mu_sell, sigma_sell, size=T)
    
    # check that s_t < c_ t
    for t in range(T):
        if p_buy[t] <= p_sell[t]:
            p_sell[t] = 0.95*p_buy[t]
    
    if False:
        x_sell = np.linspace(mu_sell - 3*sigma_sell, mu_sell + 3*sigma_sell, 100)
        x_buy = np.linspace(mu_buy - 3*sigma_buy, mu_buy + 3*sigma_buy, 100)
        plt.figure()
        plt.plot(x_sell, stats.norm.pdf(x_sell, mu_sell, sigma_sell))
        plt.plot(x_buy, stats.norm.pdf(x_buy, mu_buy, sigma_buy))
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.savefig("prices_distribution", bbox_inches='tight')
    
    i = Instance()
    dist_buying = 0
    dist_selling = 0
    i.input_param(T, delta, p_buy_min, p_buy_max, dist_buying, p_sell_min, \
                p_sell_max, dist_selling, energy, demand, capacity, \
                seed = seed, initial_battery = s0)
        
    if save:
        root_name = "Instances/I_N" + str(T) + "_C"+ str(capacity) + "_seed" + str(seed) + other_attributes
        file_name = root_name
        
        counter = 0
        while os.path.exists(file_name):
            counter = counter + 1
            file_name = root_name + "(" + str(counter) + ")"      
            
        f = open(file_name, "a")
        f.write(str(T)+"\n")
        f.write(str(delta)+"\n")
        f.write(str(p_buy_min)+"\n")
        f.write(str(p_buy_max)+"\n")
        f.write(str("uniform")+"\n")
        f.write(str(p_sell_min)+"\n")
        f.write(str(p_sell_max)+"\n")
        f.write(str("uniform")+"\n" )
        f.close()
        write_long_line(file_name, energy)
        f.close()
        write_long_line(file_name, demand)
        f = open(file_name, "a")
        f.write(str(capacity)+"\n")
        f.write(str(seed)+"\n")
        f.write(str(s0)+'\n')
        f.close()
        write_long_line(file_name, cutoff(p_buy, 2))
        f.close()
        write_long_line(file_name, cutoff(p_sell, 2))
    
    return i    

# Function to cut off at a given number of digits
def cutoff(X, digits=2):
    if np.isscalar(X):
        return round(X*(10**digits))/(10**digits)
    else:
        cut_x = np.zeros(len(X))
        for i in range(len(X)):
            cut_x[i] = cutoff(X[i], digits=digits)
        return cut_x
    
    
def convert2float(line_file, separator = ","):
    temp_line_array = line_file.split(separator)
    if len(temp_line_array) == 1:
        return(float(line_file))
    temp_float_array = np.zeros(len(temp_line_array))
    for i in range(len(temp_line_array)-1):
        temp_float_array[i] = float(temp_line_array[i])
    return temp_float_array

def convert2int(line_file, separator = ", "):
    temp_line_array = line_file.split(separator)
    if len(temp_line_array) == 1:
        return(int(line_file))
    temp_int_array = np.zeros(len(temp_line_array))
    for i in range(len(temp_line_array)-1):
        temp_int_array[i] = int(temp_line_array[i])
    return temp_int_array

def write_long_line(file_name, arr):
    f = open(file_name, "a")
    for i in range(len(arr)-1):
        f.write(str(arr[i])+", ")
    f.write(str(arr[len(arr)-1]))
    f.write("\n")
    f.close()
    
def draw_phase():
    T = 30
    C = 10
    t = range(T)
    storage = [ 6.54,  6.54, 10.  ,  6.66,  6.66, 10.  , 4.21, 10.  ,  3.  ,\
            8.23,  9.86,  9.86, 10.  ,  3.  ,  3.  ,  3.  ,  3.  ,  3.  ,\
            8.08, 10.  ,  3.  , 10.  ,  9.11, 10.  ,  3.  ,  7.5 ,  7.5 ,\
           10.  ,  3.  , 10.  ]
    fig, axs = plt.subplots(2)
    axs[0].plot(t, C*np.ones(T))
    plt.ylabel("Battery")
    axs[0].add_patch(Rectangle((12,0), 5, 10, facecolor = (100/486,149/486,237/486, 0.7)))
    axs[0].set_title("Conservative behaviour")
    # plt.plot(t, C*np.ones(T))
    # plt.ylim(-0.2,C+0.2)
    # axs[1] = fig.add_axes([0,0,1,1])
    axs[1].plot(range(T), storage, color = "black")
    plt.plot(range(T), C*np.ones(T))
    axs[1].add_patch(Rectangle((12,0), 5, 10, facecolor = (100/486,149/486,237/486, 0.7)))
    axs[1].set_title("More elaborate strategy")
    for ax in axs.flat:
        ax.set(xlabel='Days', ylabel='Battery level [Kw]')
        ax.label_outer()
    plt.tight_layout()
    # fig = plt.figure(2)
    # plt.plot(t, C*np.ones(T))
    # plt.ylim(-0.2,C+0.2)
    # ax = fig.add_axes([0,0,1,1])
    # plt.plot(range(T), storage, color = "black")
    # plt.plot(range(T), C*np.ones(T))
    # plt.xlabel("Days")
    # plt.ylabel("Battery level [Kw]")
    # ax.add_patch(Rectangle((12,0), 5, 10, facecolor = (100/486,149/486,237/486, 0.7)))


