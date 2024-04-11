import numpy as np

def random_search(x_init, x_range, f_x, max_iter):

    # initialization of variables
    x = x_init; x_visited = [x]; x_best = [x]
    f = f_x(x)

    # random search
    (x_min,x_max) = x_range
    x_arr = [(np.random.uniform(x_min,x_max), np.random.uniform(x_min,x_max), np.random.uniform(x_min,x_max)) for _ in range(max_iter)]
    x_arr = [np.round(x,3) for x in x_arr]
    for i in range(max_iter):
        x_tmp = x_arr[i]
        f_tmp = f_x(x_tmp)
        if f_tmp < f:
            x = x_tmp
            f = f_tmp
        x_visited.append(x_tmp)
        x_best.append(x)
            
    return (x,f,x_visited,x_best)

