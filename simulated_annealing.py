import numpy as np

def simulated_annealing(x_init, x_range, f_x, T_schedule, threshold, max_iter, M):

    def evaluate_neighbour(x, x_range, max_shift):
    
        (x_min, x_max) = x_range
        x_shift = tuple([max_shift*np.random.uniform(-1,1) for _ in range(3)])
        x_next = np.clip(np.round(x+x_shift,decimals=3), x_min, x_max)

        return x_next

    # initialization of variables
    x = x_init; x_min = x;  x_visited = [x]; x_best = [x]
    f = f_x(x); f_min = f
    k = 0
    T = T_schedule[k]
    max_shift = 0.5

    # simulated annealing
    while(T > threshold and k < max_iter):
        for _ in range(M):
            x_tmp = evaluate_neighbour(x, x_range, max_shift)
            f_tmp = f_x(x_tmp)

            delta = f_tmp-f
            if delta <= 0:
                x = x_tmp; f = f_tmp
            else:
                p = np.random.rand()
                T = T_schedule[k]
                if p <= np.e**(-delta/T):
                    x = x_tmp
                    f = f_tmp
            if f < f_min:
                x_min = x
                f_min = f
            x_visited.append(x_tmp)
            x_best.append(x_min)
        k += 1

    return (x_min,f_min,x_visited,x_best)

