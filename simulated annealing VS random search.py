import numpy as np
import matplotlib.pyplot as plt
from simulated_annealing import simulated_annealing
from random_search import random_search

# objective function
def f_x(x):

    (x1,x2,x3) = x
    f = 4/3*(x1**2+x2**2-x1*x2)**0.75+x3
    return f

# ---main---

x_min = 0
x_max = 2

# Simulated annealing

# determining the range of the function values
x_samples = np.linspace(x_min, x_max, 100)
X1, X2, X3 = np.meshgrid(x_samples, x_samples, x_samples, indexing='ij')
x_samples = np.vstack((X1.flatten(), X2.flatten(), X3.flatten())).T
f_values = np.array([f_x(x) for x in x_samples])
f_min_rough_est = np.min(f_values)
f_max_rough_est = np.max(f_values)
f_range_est = f_max_rough_est-f_min_rough_est

# initialization of variables
x_init = np.random.uniform(x_min, x_max, size=(3,))
x_init = np.round(x_init, decimals=3)

max_iter = 100
M = 20
T_init = 0.1*f_range_est
alpha = 0.8
T_schedule =  [alpha**(i)*T_init for i in range(max_iter)]
threshold = 1e-2

# function minimization
(x, f_min, x_visited, x_best) = simulated_annealing(x_init, (x_min,x_max), f_x, T_schedule, threshold, max_iter, M)
f_visited = [np.round(f_x(x), decimals=3) for x in x_visited]
f_best = [np.round(f_x(x), decimals=3) for x in x_best]
f_min = np.round(f_best[-1],3)

# display of results
plt.figure()
plt.subplot(1,2,1)
plt.plot(T_schedule); plt.grid(alpha=0.5)
plt.xlabel('k'); plt.ylabel('T'); plt.title('Cooling schedule, cooling rate = {}'.format(alpha))
plt.subplot(1,2,2)
plt.plot(f_visited, label = "f_current")
plt.plot(f_best, label = "f_best")
plt.xlabel('# candidates'); plt.ylabel('f_min')
plt.title('Simulated annealing, f_min = {}'.format(f_min))
plt.legend(loc='upper right'); plt.ylim((-0.1,max(f_visited))); plt.grid()

# performing simulated annealing for different cooling profiles
alpha_arr = [0.2, 0.8, 0.99]

N = 100
x_init_arr = [np.round(np.random.uniform(x_min,x_max, size=(3,)),decimals=3) for _ in range(N)]

N_mean_arr = np.zeros((4,)); N_std_arr = np.zeros((4,))
f_mean_arr = np.zeros((4,)); f_std_arr = np.zeros((4,))

plt.figure()
for i in range(3):

    alpha = alpha_arr[i]
    T_schedule = [alpha**(i)*T_init for i in range(max_iter)]

    f_visited_list = []; f_best_list = []
    N_candidates = np.zeros((N,))
    f_min_arr = np.zeros((N,))

    for j in range(N):
        (x, f_min, x_visited, x_best) = simulated_annealing(x_init_arr[j], (x_min,x_max), f_x, T_schedule, threshold, max_iter, M)
        f_visited = [np.round(f_x(x), decimals=3) for x in x_visited]
        f_best = [np.round(f_x(x), decimals=3) for x in x_best]
        f_min = np.round(f_best[-1],3)

        f_visited_list.append(f_visited)
        f_best_list.append(f_best)
        
        N_candidates[j] = np.where(f_visited == f_min)[0][0] + 1
        f_min_arr[j] = f_min
    
    f_current_avg = np.mean(np.array(f_visited_list),axis=0)
    f_best_avg = np.mean(np.array(f_best_list),axis=0)

    N_mean_arr[i+1] = np.round(np.mean(N_candidates),3)
    N_std_arr[i+1] = np.round(np.std(N_candidates),3)
    
    f_mean_arr[i+1] = np.round(np.mean(f_min_arr),3)
    f_std_arr[i+1] = np.round(np.std(f_min_arr),3)
    
    # display of results
    plt.subplot(4,3,1+i)
    plt.plot(T_schedule); plt.grid(alpha=0.5)
    plt.xlabel('k'); plt.ylabel('T'); plt.title('Cooling schedule, cooling rate = {}'.format(alpha))
    plt.subplot(4,3,4+i)
    plt.plot(f_visited, label = "f_current")
    plt.plot(f_best, label = "f_best")
    plt.xlabel('N candidates'); plt.ylabel('f_min')
    plt.title('Simulated annealing, f_mean = {}'.format(f_mean_arr[i+1]))
    plt.legend(loc='upper right'); plt.ylim((-0.1,max(f_visited))); plt.grid()
    plt.subplot(4,3,7+i)
    plt.hist(N_candidates, edgecolor='black', bins= int(np.sqrt(N))); plt.grid()
    plt.xlabel('N candidates'); plt.title('N_mean = {}, N_std = {}'.format(N_mean_arr[i+1],N_std_arr[i+1]))
    plt.subplot(4,3,10+i)
    plt.hist(f_min_arr, edgecolor='black', bins= int(np.sqrt(N)), ); plt.grid()
    plt.xlabel('f_min'); plt.title('f_mean = {}, f_std = {}'.format(f_mean_arr[i+1], f_std_arr[i+1]))
plt.tight_layout()


# Random search

x_min = 0
x_max = 2

f_visited_list = []; f_best_list = []

N_candidates = np.zeros((N,))
f_min_arr = np.zeros((N,))

for i in range(N):
    (x,f_min,x_visited,x_best) = random_search(x_init_arr[i], (x_min,x_max), f_x, max_iter*M)
    f_visited = [np.round(f_x(x), decimals=3) for x in x_visited]; f_visited_list.append(f_visited)
    f_best = [np.round(f_x(x), decimals=3) for x in x_best]; f_best_list.append(f_best)
    f_min = np.round(f_min, decimals=3)
    
    N_candidates[i] = np.where(f_visited == f_min)[0][0] + 1
    f_min_arr[i] = f_min

f_visited_avg = np.mean(np.array(f_visited_list),axis=0)
f_best_avg = np.mean(np.array(f_best_list),axis=0)

N_mean_arr[0] = np.round(np.mean(N_candidates),3)
N_std_arr[0] = np.round(np.std(N_candidates),3)
f_mean_arr[0] = np.round(np.mean(f_min_arr),3)
f_std_arr[0] = np.round(np.std(f_min_arr),3)

# display of results
plt.figure()
plt.subplot(1,3,1)
plt.plot(f_visited_avg, label = "f_current")
plt.plot(f_best_avg, label = "f_best")
plt.xlabel('N candidates'); plt.ylabel('f_min')
plt.title('Random search')
plt.legend(); plt.ylim((-0.1,max(f_visited_avg))); plt.grid()
plt.subplot(1,3,2)
plt.hist(N_candidates, edgecolor='black', bins= int(np.sqrt(N))); plt.grid()
plt.xlabel('N_candidates'); plt.title('Histogram of N_candidates, mean = {}, std = {}'.format(N_mean_arr[0],N_std_arr[0]))
plt.subplot(1,3,3)
plt.hist(f_min_arr, edgecolor='black', bins= int(np.sqrt(N))); plt.grid()
plt.xlabel('f_min'); plt.title('Histogram of f_min, mean = {}, std = {}'.format(f_mean_arr[0], f_std_arr[0]))
plt.tight_layout()


# Tabular representation of results

table_data = [N_mean_arr, N_std_arr, f_mean_arr, f_std_arr]
row_labels = ['N_mean','N_std','f_mean','f_std']

# display of results
fig, ax = plt.subplots()
table = ax.table(cellText=table_data, loc='center', colLabels=['random search', 'simulated annealing (cooling rate = 0.2)',\
            'simulated annealing (cooling rate = 0.8)','simulated annealing (cooling rate = 0.99)'],\
                rowLabels=row_labels, cellLoc='center', colColours=['#f2f2f2']*4)
ax.set_title('Search results'); ax.set_axis_off()
table.auto_set_font_size(False); table.set_fontsize(10)
table.scale(1.2, 1.2)

plt.show()

