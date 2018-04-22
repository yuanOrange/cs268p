import numpy as np
import time


counter = 0


def fx_one(x):
    global counter
    counter += 1
    return x * x - 2 * x + 1


def fx_two(x):
    global counter
    counter += 1
    return abs(x+1)


def fx_three(x):
    global counter
    counter += 1
    return np.exp(x+1) + np.exp(-x)


def fx_four(x):
    global counter
    counter += 1
    return np.sin(0.2 * x + 5)


def optimizer(fx, x1, s):
    tau = 0.618
    epsl_r = 1.49e-8
    epsl_x = 1e-4
    epsl_f = 1e-6
    f1 = fx(x1)
    x2 = x1 + s
    f2 = fx(x2)

    if f2 > f1:
        temp = x1
        x1 = x2
        x2 = temp
        s = -s

    while True:
        s = s/tau
        x4 = x2 + s
        f4 = fx(x4)

        if f4 > f2:
            break

        x1, f1 = x2, f2
        x2, f2 = x4, f4

    f_old = (f1+f2+f4)/3

    x3 = tau * x4 + (1 - tau) * x1
    f3 = fx(x3)
    if f2 < f3:
        x4, f4 = x1, f1
        x1, f1 = x3, f3
    else:
        x1, f1 = x2, f2
        x2, f2 = x3, f3

    f_avg = (f1+f2+f4)/3

    while (abs(x1-x4) > epsl_r * abs(x2) + epsl_x) or (abs(f_avg-f_old) > epsl_r * abs(f2) + epsl_f):
        f_old = f_avg

        x3 = tau * x4 + (1-tau) * x1
        f3 = fx(x3)

        if f2 < f3:
            x4, f4 = x1, f1
            x1, f1 = x3, f3
        else:
            x1, f1 = x2, f2
            x2, f2 = x3, f3

        f_avg = (f1+f2+f4)/3

    return x2, f2


def stat(l):
    return np.mean(l), (np.std(l)*np.sqrt(1000)/np.sqrt(1000-1))


def multi_test(fx):

    x_list = []
    f_list = []
    t_list = []
    c_list = []
    global counter

    for i in range(1, 1000):
        x_start = np.random.uniform(-10, 10)
        s_start = np.random.random()
        counter = 0
        start = time.time()
        x, f = optimizer(fx, x_start, s_start)
        end = time.time()
        t = (end - start)*1000
        x_list.append(x)
        f_list.append(f)
        t_list.append(t)
        c_list.append(counter)

    x_avg, x_std = stat(x_list)
    f_avg, f_std = stat(f_list)
    t_avg, t_std = stat(t_list)
    c_avg, c_std = stat(c_list)

    print("\nx_min:{0:.4f}+{1:.4f}".format(x_avg, x_std))
    print("f_min:{0:.4f}+{1:.4f}".format(f_avg, f_std))
    print("t_min:{0:.4f}+{1:.4f}".format(t_avg, t_std))
    print("c_min:{0:.4f}+{1:.4f}".format(c_avg, c_std))


multi_test(fx_one)
multi_test(fx_two)
multi_test(fx_three)
multi_test(fx_four)
