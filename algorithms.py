# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 13:14:30 2023
Algorithms from paper
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB

import heapq


def IP(i, P = 0, time_limit = 5*60):
    m = gp.Model("Competitor") # "Prosumer Energy Problem"
    diff = i.demand - i.energy # d'_{t}
    if time_limit != False:
        m.setParam('TimeLimit', time_limit)
    
    # Variables
    x = m.addVars(i.T, vtype=GRB.INTEGER, lb=0.0, obj=i.p_buy, name="x")
    z = m.addVars(i.T, vtype=GRB.INTEGER, lb=-int(i.capacity), name = "z")
    SOC = m.addVars(i.T, vtype=GRB.INTEGER, lb = 0.0, ub = i.capacity, name="SOC")
    
    # Variables for energy sold y and balance of energy for problems with selling
    if(P % 2 == 1):
        y = m.addVars(i.T, vtype=GRB.INTEGER, lb=0.0, obj = -i.p_sell, name="y")
        m.addConstrs((x[t] - y[t] - z[t] == diff[t] for t in range(i.T)), "demand")
    else:
        m.addConstrs((x[t] - z[t] == diff[t] for t in range(i.T)), "demand")
    
    # Bounds on energy bought
    if(P==2 or P==3):
        m.addConstrs((x[t] <= max(0,diff[t]) for t in range(i.T)), name = "buy_at_most")
    
    # Contrains on energy bought
    if(P == 4 or P == 5):
        q = m.addVars(i.T, vtype=GRB.BINARY, name="q")
        m.addConstrs((x[t] == diff[t]*q[t] for t in range(i.T)), "bound_buy")

    # Update of the state of the system SOC
    m.addConstrs((SOC[t] - z[t] == SOC[t-1] for t in range(1,i.T)), name = "capacity")
    m.addConstr(SOC[0] - z[0] == i.initial_battery, name = "initial_capacity")
    
    m.ModelSense = GRB.MINIMIZE # objective function
    # m.write('comparison.lp') # txt file with the LP problem
    m.Params.LogToConsole = 0 # suppress printing on console while executing
    m.optimize()
    Z = np.zeros(i.T)
    Soc = np.zeros(i.T)
    x_p = np.zeros(i.T)
    y_p = np.zeros(i.T)
    cost = 0
    
    status = m.status 
    if status == GRB.INFEASIBLE:
#        print("The model is infeasible")
        return cost, x_p, y_p, Soc, Z, -1
    if status == GRB.UNBOUNDED:
#        print("The model is unbounded")
        return cost, x_p, y_p, Soc, Z, -1
    if status == GRB.INF_OR_UNBD:
#        print("The model is infeasible or unbounded")
        return cost, x_p, y_p, Soc, Z, -1
    if status == GRB.OPTIMAL:
        cost = m.ObjVal
#        print("Optimal")
        for idx in range(i.T):
            Z[idx] = z[idx].X
            Soc[idx] = SOC[idx].X
            x_p[idx] = x[idx].X
            if(P % 2 == 1):
                y_p[idx] = y[idx].X
        return cost, x_p, y_p, Soc, Z, 1
    if status == GRB.TIME_LIMIT:
        return cost, x_p, y_p, Soc, Z, 2


def P1(i):
    costs = 0
    x = np.zeros(i.T)
    f = 0
    SOC = np.zeros(i.T+1)
    SOC[0] = i.initial_battery
    R = np.zeros(i.T+1)
    R[0] = i.capacity - SOC[0]
    alpha = i.capacity*np.ones(i.T)
    c = i.p_buy.copy()
    min_c = 0
    
    
    for t in range(i.T):
        diff = i.demand[t] - i.energy[t]
        if c[t]<=c[min_c]:
            min_c = t
        i_star = min_c
#        i_star = np.argmin(c[f:(t+1)]) + f
#        print("Iter "+str(t)+", i_star = "+str(i_star)+", min_c = "+str(min_c))
        if diff < 0: # in case I have extra energy from PV I just update time t+1
            SOC[t+1] = SOC[t] - diff
            R[t+1] = R[t] + diff
            for j in range((i_star),(t+1)):
                R[j] = min(R[j], R[t+1])
        elif SOC[t] >= diff: # in case I need energy but I can use the one I have in the battery
            SOC[t+1] = SOC[t] - diff
            R[t+1] = R[t] + diff
            alpha[t] = alpha[t] + diff
        else:
            R[t+1] = i.capacity + diff
            diff = diff - SOC[t]
            alpha[t] = alpha[t] + diff
            while diff > 0:
                incr = min(R[i_star+1], alpha[i_star] - x[i_star], diff)
                costs = costs + incr*c[i_star]
                x[i_star] = x[i_star] + incr
                for j in range(t, i_star, -1):
                    SOC[j] = SOC[j] + incr
                    R[j] = R[j] - incr
                    if R[j] == 0:
                        f = j
#                    for j in range(f+1, i_star+1):
#                        R[j] = min(R[j], R[i_star+1])
#                    if x[i_star] >= alpha[i_star]:
#                        c[i_star] = float('inf')
                diff = diff - incr
                i_star = np.argmin(c[f:(t+1)]) + f
                min_c = i_star
            R[t+1] = i.capacity
            SOC[t+1] = 0
        if SOC[t+1]>i.capacity: # the instance is infeasible
            return costs, x, x, SOC, R, -1
        if SOC[t+1]==i.capacity:
            f = t
    return costs, x, x, SOC, R, 1


def P2(i):
    T = 2*i.T
    c = np.zeros(T)
    dem = np.zeros(T)
    en = np.zeros(T)
    alpha = np.zeros(T)
    betas = np.zeros(i.T)
#    s = i.p_sell.copy()
#    min_c = 0
    costs = 0
    x = np.zeros(2*i.T)
    f = 0
    SOC = np.zeros(2*i.T+1)
    SOC[0] = i.initial_battery
    R = np.zeros(2*i.T+1)
    R[0] = i.capacity - SOC[0]
#    alpha = i.capacity*np.ones(i.T)
#    min_c = 0
    x_result = np.zeros(i.T)
    y_result = np.zeros(i.T)
    cost_result = 0
    current_costs = []
    heapq.heapify(current_costs)
    i_star = 0
    
    for t in range(T):
        if (t % 2) == 0:
            c[t] = i.p_buy[(t+1)//2]
            dem[t] = max(0,i.demand[(t+1)//2]-i.energy[(t+1)//2])
            alpha[t] = i.capacity + dem[t]
        else:
            c[t] = i.p_sell[t//2]            
            en[t] = max(0, i.energy[t//2] - i.demand[t//2])
            betas[t//2] = i.capacity - max(0, i.demand[t//2]-i.energy[t//2])+en[t]
#            betas[t//2] = i.capacity +en[t]
#            betas[t//2] = i.capacity + max(en[t], i.demand[t//2]-i.energy[t//2])
            dem[t] = betas[t//2]
            alpha[t] = betas[t//2]
            y_result[t//2] = betas[t//2]
            cost_result = cost_result - betas[t//2]*i.p_sell[t//2]
#        if c[t] <= c[min_c]:
#            min_c = t
#        i_star = min_c
#        i_star = np.argmin(c[f:(t+1)]) + f

        heapq.heappush(current_costs, (c[t],t))
        i_star = current_costs[0][1]
        diff = dem[t] - en[t]
        if diff < 0: # in case I have extra energy from PV I just update time t+1
            SOC[t+1] = SOC[t] - diff
            R[t+1] = R[t] + diff
            for j in range((f+1),(t+1)):
                R[j] = min(R[j], R[t+1])
        elif SOC[t] >= diff: # in case I need energy but I can use the one I have in the battery
            SOC[t+1] = SOC[t] - diff
            R[t+1] = R[t] + diff
        else:
            R[t+1] = i.capacity + diff
            diff = diff - SOC[t]
            while diff > 0:
                full_battery = 0
                incr = min(R[i_star+1], alpha[i_star] - x[i_star], diff)
                costs = costs + incr*c[i_star]
                cost_result = cost_result + incr*c[i_star]
                x[i_star] = x[i_star] + incr
                if i_star%2 == 0:
                    x_result[i_star//2] = x_result[(i_star)//2] + incr
                else:
                    y_result[(i_star)//2] = y_result[(i_star)//2] - incr
                for j in range(i_star+1, t+1):
                    SOC[j] = SOC[j] + incr
                    R[j] = R[j] - incr
                    if R[j] == 0:
                        f = j
                        full_battery = 1
                for j in range(f, i_star+1):
                    R[j] = min(R[j], R[i_star+1])
                if x[i_star] >= alpha[i_star] and full_battery!=1:
#                    c[i_star] = float('inf')
##                        update_i_star 
#                    if len(current_costs)>0:
##                        while current_costs[0][1]<f:
#                    heapq.heappop(current_costs)
#                    i_star = current_costs[0][1]
                    heapq.heappop(current_costs)
                    i_star = current_costs[0][1]
                    while i_star < f:
                        if len(current_costs)>0:
                            heapq.heappop(current_costs)
                            i_star = current_costs[0][1]
                        else:
                            i_prime = t+1
                if full_battery:
                    i_prime = max(f-1,0)
                    while i_prime < f:
                        if len(current_costs)>0:
                            heapq.heappop(current_costs)
                            i_prime = current_costs[0][1]
                        else:
                            i_prime = t
                    i_star = i_prime
                diff = diff - incr                    
            R[t+1] = i.capacity
            SOC[t+1] = 0
        if SOC[t+1]>i.capacity: # the instance is infeasible
            return costs, x, x, SOC, R, -1
        if SOC[t+1]==i.capacity:
            f = t
    
#    for t in range(i.T):
#        x_result[t] = x[2*t]
#        y_result[t] = betas[t]-x[2*t+1]        
    
    return cost_result, x_result, y_result, SOC, R, 1


def P3(i):
    # istanza che non mi viene t = 25 seed = 48913 capacity = 403 penso che il problema sia il while 
    costs = 0
    x = np.zeros(i.T)
    f = 0
    SOC = np.zeros(i.T+1)
    SOC[0] = i.initial_battery
    R = np.zeros(i.T+1)
    R[0] = i.capacity - SOC[0]
    alpha = np.zeros(i.T)
    c = i.p_buy.copy()
#    min_c = 0
    current_costs = []
    heapq.heapify(current_costs)
    i_star = 0
    
    
    for t in range(i.T):
        diff = i.demand[t] - i.energy[t]
        alpha[t] = alpha[t] + max(0, diff)
#        i_star = np.argmin(c[f:(t+1)]) + f
#        if c[t]<=c[min_c]:# and alpha[t] != 0:
#            min_c = t
#        i_star = min_c
        if alpha[t]>0:
            heapq.heappush(current_costs, (c[t],t))
            i_star = current_costs[0][1]
        if diff < 0: # in case I have extra energy from PV I just update time t+1
            SOC[t+1] = SOC[t] - diff
            R[t+1] = R[t] + diff
            for j in range((i_star),(t+1)):
                R[j] = min(R[j], R[t+1])
        elif SOC[t] >= diff: # in case I need energy but I can use the one I have in the battery
            SOC[t+1] = SOC[t] - diff
            R[t+1] = R[t] + diff
        else:
            R[t+1] = i.capacity + diff
            diff = diff - SOC[t]
            while diff > 0:
#                update_i_star = 0
                full_battery = 0
                incr = min(R[i_star+1], alpha[i_star] - x[i_star], diff)
                costs = costs + incr*c[i_star]
                x[i_star] = x[i_star] + incr
                for j in range(i_star+1, t+1):
                    SOC[j] = SOC[j] + incr
                    R[j] = R[j] - incr
                    if R[j] == 0:
                        f = j 
                        full_battery = 1
                for j in range(f+1, i_star+1):
                    R[j] = min(R[j], R[i_star+1])
                if x[i_star] >= alpha[i_star] and full_battery!=1:
#                    c[i_star] = float('inf')
#                        update_i_star 
                    heapq.heappop(current_costs)
                    i_star = current_costs[0][1]
                    while i_star < f:
                        if len(current_costs)>0:
                            heapq.heappop(current_costs)
                            i_star = current_costs[0][1]
                        else:
                            i_prime = t+1
                if full_battery:
                    i_prime = max(f-1,0)
                    while i_prime < f:
                        if len(current_costs)>0:
                            heapq.heappop(current_costs)
                            i_prime = current_costs[0][1]
                        else:
                            i_prime = t
                    i_star = i_prime
                diff = diff - incr
#                if update_i_star:
#                    i_star = np.argmin(c[f:(t+1)]) + f
#                    min_c = i_star
            R[t+1] = i.capacity
            SOC[t+1] = 0
        if SOC[t+1]>i.capacity: # the instance is infeasible
            return costs, x, x, SOC, R, -1
        if SOC[t+1]==i.capacity:
            f = t
    return costs, x, x, SOC, R, 1


def P4(i):
    T = 2*i.T
    c = np.zeros(T)
    dem = np.zeros(T)
    en = np.zeros(T)
    alpha = np.zeros(T)
    betas = np.zeros(i.T)
    s = i.p_sell.copy()
    min_c = 0
    costs = 0
    x = np.zeros(2*i.T)
    f = 0
    SOC = np.zeros(2*i.T+1)
    SOC[0] = i.initial_battery
    R = np.zeros(2*i.T+1)
    R[0] = i.capacity - SOC[0]
#    alpha = i.capacity*np.ones(i.T)
    min_c = 0
    x_result = np.zeros(i.T)
    y_result = np.zeros(i.T)
    cost_result = 0
    current_costs = []
    heapq.heapify(current_costs)
    i_star = 0
    
    for t in range(T):
        if (t % 2) == 0:
            c[t] = i.p_buy[(t+1)//2]
            dem[t] = max(0,i.demand[(t+1)//2]-i.energy[(t+1)//2])
            alpha[t] = dem[t]
        else:
            c[t] = i.p_sell[t//2]            
            en[t] = max(0, i.energy[t//2] - i.demand[t//2])
            betas[t//2] = i.capacity - max(0, i.demand[t//2]-i.energy[t//2])+en[t]
#            betas[t//2] = i.capacity +en[t]
#            betas[t//2] = i.capacity + max(en[t], i.demand[t//2]-i.energy[t//2])
            dem[t] = betas[t//2]
            alpha[t] = betas[t//2]
            y_result[t//2] = betas[t//2]
            cost_result = cost_result - betas[t//2]*i.p_sell[t//2]
#        if c[t] <= c[min_c] and alpha[t]>0:
#            min_c = t
#        i_star = min_c
        if alpha[t]>0:
            heapq.heappush(current_costs, (c[t],t))
            i_star = current_costs[0][1]
#        i_star = np.argmin(c[f:(t+1)]) + f
        diff = dem[t] - en[t]
        if diff < 0: # in case I have extra energy from PV I just update time t+1
            SOC[t+1] = SOC[t] - diff
            R[t+1] = R[t] + diff
            for j in range((i_star),(t+1)):
                R[j] = min(R[j], R[t+1])
        elif SOC[t] >= diff: # in case I need energy but I can use the one I have in the battery
            SOC[t+1] = SOC[t] - diff
            R[t+1] = R[t] + diff
#            alpha[t] = alpha[t] + diff[t]
        else:
#            alpha[t] = alpha[t] + diff[t]
            R[t+1] = i.capacity + diff
            diff = diff - SOC[t]
            while diff > 0:
                full_battery = 0
                incr = min(R[i_star+1], alpha[i_star] - x[i_star], diff)
#                incr = min(R[i_star+1], alpha[i_star] - x[i_star], diff)
                costs = costs + incr*c[i_star]
                cost_result = cost_result + incr*c[i_star]
                x[i_star] = x[i_star] + incr
                if i_star%2 == 0:
                    x_result[i_star//2] = x_result[(i_star)//2] + incr
                else:
                    y_result[(i_star)//2] = y_result[(i_star)//2] - incr
                for j in range(t, i_star, -1):
                    SOC[j] = SOC[j] + incr
                    R[j] = R[j] - incr
                    if R[j] == 0:
                        f = j
                        full_battery = 1
                for j in range(f+1, i_star+1):
                    R[j] = min(R[j], R[i_star+1])
                if x[i_star] >= alpha[i_star] and full_battery!=1:
#                    c[i_star] = float('inf')
#                        update_i_star 
                    heapq.heappop(current_costs)
                    i_star = current_costs[0][1]
                    while i_star < f:
                        if len(current_costs)>0:
                            heapq.heappop(current_costs)
                            i_star = current_costs[0][1]
                        else:
                            i_prime = t+1
                if full_battery:
                    i_prime = max(f-1,0)
                    while i_prime < f:
                        if len(current_costs)>0:
                            heapq.heappop(current_costs)
                            i_prime = current_costs[0][1]
                        else:
                            i_prime = t
                    i_star = i_prime
                diff = diff - incr
#                if update_i_star:
#                    i_star = np.argmin(c[f:(t+1)]) + f
#                    min_c = i_star
            R[t+1] = i.capacity
            SOC[t+1] = 0
        if SOC[t+1]>i.capacity: # the instance is infeasible
            return costs, x, x, SOC, R, -1
        if SOC[t+1]==i.capacity:
            f = t
    
#    for t in range(i.T):
#        x_result[t] = x[2*t]
#        y_result[t] = betas[t]-x[2*t+1]        
    
    return cost_result, x_result, y_result, SOC, R, 1


def P5(i):
    C = int(i.capacity)
    diff = i.demand - i.energy
    R = float('-inf')*np.ones(C+1)
    R[int(i.initial_battery)] = 0
    x = np.zeros(i.T)
    y = np.zeros(i.T)
    
    for t in range(i.T):
        Q = float('-inf')*np.ones(C+1)
        if diff[t] >= 0:
            ex = 0
            d_prime = int(diff[t])
        else:
            ex = -int(diff[t])
            d_prime = 0
        alpha = d_prime
        for k in range(C+1):
            A = float('-inf')
            if C-k-ex+d_prime >= 0 and C-k-ex+d_prime <= C:
                if R[C-k-ex+d_prime ] != float('inf'):
                    value = R[C-k-ex+d_prime ]
                    if value > A:
                        A = value
            B = float('-inf')
            if C-k-ex+d_prime-alpha >= 0 and C-k-ex+d_prime-alpha <= C:
                if R[C-k-ex+d_prime-alpha] != float('inf'):
                    value = R[C-k-ex+d_prime-alpha]-alpha*i.p_buy[t]
                    if value>B:
                        B = value
            if A >= B:
                Q[C-k] = A
            else:
                Q[C-k] = B
                x[t] = d_prime
        R = Q.copy()
    feas = 1
    costs = max(R)
    if costs == float('-inf'):
        feas = -1
        costs = 0
    return(-costs), x, y, 1, 1, feas

    
def P6(i):
#    seed 11154 C 10 T 10
    C = int(i.capacity)
    diff = i.demand - i.energy
    R = -np.inf* np.ones(C+1)
    R[int(i.initial_battery)] = 0
    Q = np.zeros(C+1)
       
    for t in range(i.T):
        ex =  max(-diff[t],0)
        d_p = max(diff[t],0)
        A = -np.inf*np.ones(C+1)
        B = -np.inf*np.ones(C+1)
        for k in range(C+1):
            a1 = int(k + ex -d_p)
            a0 = int(max(0,k+ex-d_p-C))
           
            if a1>a0+1 and k>0:
                if a0==0:
                    A[C-k] = max(A[C-k+1]+i.p_sell[t], R[C-a1])
                else:
                    A[C-k] = A[C-k+1] + i.p_sell[t]
            else:
                for y in range(a0,a1+1):
                    if R[C+y -a1] + y*i.p_sell[t]>A[C-k]:
                        A[C-k] = R[C+y-a1] + y*i.p_sell[t]
           
            b0 = int(max(0,k+ex-C))
            b1 = int(k + ex)
            if b1>b0+1 and k>0:
                if b0 == 0:
                    B[C-k] = max(B[C-k+1]+i.p_sell[t],R[C-b1]-d_p*i.p_buy[t])
                else:
                    B[C-k] = B[C-k+1]+i.p_sell[t]
            else:    
                for y in range(b0,b1+1):
                    if R[C+y-b1] + y*i.p_sell[t]-d_p*i.p_buy[t]>B[C-k]:
                        B[C-k] =  R[C+y-b1] + y*i.p_sell[t]-d_p*i.p_buy[t]
            Q[C-k] = max(A[C-k],B[C-k])
           
        R = Q[0:C+1]
       
        Q = np.zeros(C+1)
        costs = np.max(R)
        if costs > -np.inf:
            feas = 1
        else:
            feas = 0
            return -costs, 0, 0, 0,0,  feas
   
    return -costs, 0, 0, 0,0,  feas

