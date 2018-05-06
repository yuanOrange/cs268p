import numpy as np
import time


tau = 0.618
epsl_r = 1.49e-8
epsl_x = 1e-6
epsl_f = 1e-8
np.random.seed(0)


# def rosenbrock(point):
#     x = point[0]
#     y = point[1]
#     return (1-x)**2 + 100*(y-x*x)**2 + 1

def rosenbrock(x):
    res = 0
    for i in range(x.size-1):
        res += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return res + 1


# def grad_rosenbrock(point):
#     x = point[0]
#     y = point[1]
#     grad = [None]*2
#     grad[0] = 400*(x**3) - 400*x*y + 2*x - 2
#     grad[1] = -200*(x*x) + 200*y
#     return grad

def grad_rosenbrock(x):
    order = x.size
    grad = np.zeros(order)

    grad[0] = - 400 * x[1] * x[0] + 400 * x[0]**3 + 2 * x[0] - 2
    grad[order-1] = 200 * x[order-1] - 200 * x[order-2]**2

    for i in range(1, order-1):
        grad[i] = - 400 * x[i+1] * x[i] + 400 * x[i]**3 + 202 * x[i] - 200 * x[i-1]**2 - 2

    return grad


# def test1(point):
#     x = point[0]
#     y = point[1]
#     return (x-1)**2 + (y-2)**2 + x*y
#
#
# def grad_test1(point):
#     x = point[0]
#     y = point[1]
#     dirc = [None]*2
#     dirc[0] = 2 * (x-1) + y
#     dirc[1] = 2 * (y-2) + x
#     return dirc
#
#
# def test2(point):
#     x = point[0]
#     y = point[1]
#     return (x-2)**4 + (y+1)**4 - x * y
#
#
# def grad_test2(point):
#     x = point[0]
#     y = point[1]
#     dirc = [None] * 2
#     dirc[0] = 4 * (x-2)**3 - y
#     dirc[1] = 4 * (y+1)**3 - x
#     return dirc


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


# def gold_select(point1, point4):
#     point3 = [None]*2
#     point3[0] = tau * point4[0] + (1 - tau) * point1[0]
#     point3[1] = tau * point4[1] + (1 - tau) * point1[1]
#     return point3

def gold_select(x1, x2):
    x = np.zeros(x1.size)
    for i in range(x.size):
        x[i] = tau * x2[i] + (1 - tau) * x1[i]
    return x


# def distance(point1, point2):
#     return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


# def norm(point):
#     return np.sqrt((point[0])**2 + (point[1])**2)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


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

        if np.linalg.norm(point1-point4) < epsl_r * np.linalg.norm(point2) + epsl_x:
            break
        if abs(f_new - f_old) < epsl_r * abs(f2) + epsl_f:
            break

        f_old = f_new

    return point2, f2


def coord_desc(fx, point, step):

    value_0 = fx(point)
    point_0 = point
    count = 0

    while True:
        if count % 2 == 0:
            dirc = [1, 0]
        else:
            dirc = [0, 1]

        point, value = optimizer(fx, point, dirc, step)
        count = count + 1

        if abs(value - value_0) < epsl_f + epsl_r * value_0:
            break
        if np.linalg.norm(point-point_0) < epsl_x + epsl_r * np.linalg.norm(point_0):
            break
        # if count > 1e6:
        #     break

        point_0 = point
        value_0 = value

    return point, value, count


def steep_desc(fx, grad_fx, point, step):

    value_0 = fx(point)
    point_0 = point
    count = 0

    while True:
        dirc = grad_fx(point)
        dirc = - dirc

        point, value = optimizer(fx, point, dirc, step)
        count = count + 1

        if abs(value - value_0) < epsl_f + epsl_r * value_0:
            break
        if np.linalg.norm(point-point_0) < epsl_x + epsl_r * np.linalg.norm(point_0):
            break

        point_0 = point
        value_0 = value

    return point, value, count


def stat(l):
    return np.mean(l), (np.std(l)*np.sqrt(100)/np.sqrt(100-1))


def multi_test(fx, grad_fx, fp):
    p1_0_list, p2_0_list = [], []
    p1_1_list, p2_1_list = [], []
    f1_list, f2_list = [], []
    t1_list, t2_list = [], []
    c1_list, c2_list = [], []

    for i in range(1, 100):
        x_start = np.random.uniform(-5, 5)
        y_start = np.random.uniform(-5, 5)
        s_start = np.random.random()

        start = time.time()
        p1, f1, c1 = coord_desc(fx, [x_start, y_start], s_start)
        end = time.time()
        t1 = (end - start) * 1000
        p1_0_list.append(p1[0])
        p1_1_list.append(p1[1])
        f1_list.append(f1)
        t1_list.append(t1)
        c1_list.append(c1)

        start = time.time()
        p2, f2, c2 = steep_desc(fx, grad_fx, [x_start, y_start], s_start)
        end = time.time()
        t2 = (end - start) * 1000
        p2_0_list.append(p2[0])
        p2_1_list.append(p2[1])
        f2_list.append(f2)
        t2_list.append(t2)
        c2_list.append(c2)

    p1_0_avg, p1_0_std = stat(p1_0_list)
    p1_1_avg, p1_1_std = stat(p1_1_list)
    f1_avg, f1_std = stat(f1_list)
    t1_avg, t1_std = stat(t1_list)
    c1_avg, c1_std = stat(c1_list)

    p2_0_avg, p2_0_std = stat(p2_0_list)
    p2_1_avg, p2_1_std = stat(p2_1_list)
    f2_avg, f2_std = stat(f2_list)
    t2_avg, t2_std = stat(t2_list)
    c2_avg, c2_std = stat(c2_list)

    print("\np[0]_min:{0:.6f}+{1:.6f}".format(p1_0_avg, p1_0_std), file=fp)
    print("p[1]_min:{0:.6f}+{1:.6f}".format(p1_1_avg, p1_1_std), file=fp)
    print("f_min:{0:.6f}+{1:.6f}".format(f1_avg, f1_std), file=fp)
    print("t_min:{0:.6f}+{1:.6f}".format(t1_avg, t1_std), file=fp)
    print("c_min:{0:.6f}+{1:.6f}".format(c1_avg, c1_std), file=fp)
    print("\np[0]_min:{0:.6f}+{1:.6f}".format(p2_0_avg, p2_0_std), file=fp)
    print("p[1]_min:{0:.6f}+{1:.6f}".format(p2_1_avg, p2_1_std), file=fp)
    print("f_min:{0:.6f}+{1:.6f}".format(f2_avg, f2_std), file=fp)
    print("t_min:{0:.6f}+{1:.6f}".format(t2_avg, t2_std), file=fp)
    print("c_min:{0:.6f}+{1:.6f}\n".format(c2_avg, c2_std), file=fp)


fp = open('./output.txt', 'w+')
multi_test(rosenbrock, grad_rosenbrock, fp)
multi_test(test1, grad_test1, fp)
multi_test(test2, grad_test2, fp)
