import numpy as np
import time
import matplotlib.pyplot as plt

np.random.seed(int(time.time()))
tau = 0.618
epsl_r = 1.49e-8
epsl_x = 1e-4
epsl_f = 1e-6


def rosenbrock(x):
    res = 0
    for i in range(x.size-1):
        res += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return res + 1


def grad_rosenbrock(x):
    order = x.size
    grad = np.zeros(order)

    grad[0] = - 400 * x[1] * x[0] + 400 * x[0]**3 + 2 * x[0] - 2
    grad[order-1] = 200 * x[order-1] - 200 * x[order-2]**2
    for i in range(1, order-1):
        grad[i] = - 400 * x[i+1] * x[i] + 400 * x[i]**3 + 202 * x[i] - 200 * x[i-1]**2 - 2

    return grad


def test_one(x):
    return (x[0] - 1) ** 2 + (x[1] - 2) ** 2 + x[0] * x[1]


def grad_one(x):
    grad = np.zeros(2)
    grad[0] = 2 * (x[0] - 1) + x[1]
    grad[1] = 2 * (x[1] - 2) + x[0]
    return grad


def stat_scalar(l):
    return np.mean(l), (np.std(l)*np.sqrt(len(l))/np.sqrt(len(l)-1))


def stat_vector(l):
    std = 0
    length = len(l)
    order = l[0].size
    v_avg = np.zeros(order)

    for i in range(length):
        v_avg += l[i]
    v_avg = v_avg / length

    for i in range(length):
        std += (np.linalg.norm(l[i] - v_avg))
    std = std / np.sqrt(length-1)

    return v_avg, std


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def take_step(x, dirc, step):
    dirc = normalize(dirc)
    x_des = np.zeros(x.size)
    for i in range(x.size):
        x_des[i] = x[i] + step * dirc[i]
    return x_des


def gold_select(x1, x2):
    x = np.zeros(x1.size)
    for i in range(x.size):
        x[i] = tau * x2[i] + (1 - tau) * x1[i]
    return x


def optimizer(fx, point1, dirc, step):

    f1 = fx(point1)
    point2 = take_step(point1, dirc, step)
    f2 = fx(point2)

    if f2 > f1:
        point1, point2 = point2, point1
        f1, f2 = f2, f1
        dirc = - dirc

    while True:
        step = step / tau
        point4 = take_step(point2, dirc, step)
        f4 = fx(point4)

        if f4 > f2:
            break

        point1, f1 = point2, f2
        point2, f2 = point4, f4

    f_old = (f1 + f2 + f4) / 3

    while True:
        point3 = gold_select(point1, point4)
        f3 = fx(point3)

        if f2 < f3:
            point4, f4 = point1, f1
            point1, f1 = point3, f3
        else:
            point1, f1 = point2, f2
            point2, f2 = point3, f3

        f_new = (f1 + f2 + f4) / 3

        if np.linalg.norm(point4 - point1) < epsl_r * np.linalg.norm(point2) + epsl_x:
            break
        if abs(f_new - f_old) < epsl_r * abs(f2) + epsl_f:
            break

        f_old = f_new

    return point2, f2


def conj_grad(func, func_grad, x, step, cg_iter):

    x_old = x
    f_old = func(x_old)
    count = 0

    while True:

        g_old = func_grad(x_old)
        d = - g_old
        # print('restart')

        for _ in range(cg_iter):
            x_new, f_new = optimizer(func, x_old, d, step)
            count += 1
            g_new = func_grad(x_new)
            beta = g_new.dot(g_new) / g_old.dot(g_old)
            d = - g_new + beta * d

            if np.linalg.norm(x_new - x_old) < epsl_x + epsl_r * np.linalg.norm(x_old):
                return x_new, f_new, count
            if abs(f_new - f_old) < epsl_f + epsl_r * f_old:
                return x_new, f_new, count

            x_old, f_old, g_old = x_new, f_new, g_new


def multi_test(func, func_grad, dim, it):

    x_global = []
    f_global = []
    x_local = []
    f_local = []
    count_list = []

    for _ in range(100):
        x = np.zeros(dim)
        step = np.random.random()
        for i in range(dim):
            x[i] = np.random.uniform(-5, 5)
        x_temp, f_temp, count = conj_grad(func, func_grad, x, step, it)

        if func == rosenbrock and abs(x_temp[0] + 1) < 1e-2:
                x_local.append(x_temp)
                f_local.append(f_temp)
        else:
            x_global.append(x_temp)
            f_global.append(f_temp)
            count_list.append(count)

    np.set_printoptions(precision=6, suppress=True)
    t1, t2 = stat_vector(x_global)
    v1, v2 = stat_scalar(f_global)

    if func == rosenbrock:
        rel_x1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        rel_f1 = 1
    else:
        rel_x1 = np.array([0,2])
        rel_f1 = 1
    t2 = np.linalg.norm(t1-rel_x1) / np.linalg.norm(rel_x1)
    v2 = abs(v1-rel_f1) / rel_f1
    print('x_avg:{}, relative:{:.6f}'.format(t1, t2))
    print('f_avg:{:.6f}, relative:{:.6f}'.format(v1, v2))
    if func == rosenbrock:
        t1, t2 = stat_vector(x_local)
        v1, v2 = stat_scalar(f_local) 
        rel_x1 = np.array([-1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        rel_f1 = 5
        t2 = np.linalg.norm(t1 - rel_x1) / np.linalg.norm(rel_x1)
        v2 = abs(v1 - rel_f1) / rel_f1
        print('\nx_local_avg:{}, relative:{:.6f}'.format(t1, t2))
        print('f_local_avg:{:.6f}, relative:{:.6f}'.format(v1, v2))

    return count_list


iter_array = [6, 8, 10, 12, 14]
cg_count = []
cg_count_std = []
for it in iter_array:
    print('\n', it)
    c_l = multi_test(rosenbrock, grad_rosenbrock, 10, it)
    c1, c2 = stat_scalar(c_l)
    cg_count.append(c1)
    cg_count_std.append(c2)
plt.errorbar(iter_array, cg_count, cg_count_std, ecolor='g', capsize=5)
plt.show()

iter_array = [1, 2, 3, 4]
cg_count.clear()
cg_count_std.clear()
for it in iter_array:
    print('\n', it)
    c_l = multi_test(test_one, grad_one, 2, it)
    c1, c2 = stat_scalar(c_l)
    cg_count.append(c1)
    cg_count_std.append(c2)
plt.errorbar(iter_array, cg_count, cg_count_std, ecolor='g', capsize=5)
plt.show()
