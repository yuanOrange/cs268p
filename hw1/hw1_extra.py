import numpy as np
import matplotlib.pyplot as plt


counter = 0


def fx_one(x):
    return x * x - 2 * x + 1


def optimizer(fx, x1, s):
    tau = 0.618
    global epsl_r
    global epsl_x
    global epsl_f
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
        x1, x1 = x2, f2
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
    global counter

    for i in range(1, 1000):
        x_start = np.random.uniform(-10, 10)
        s_start = np.random.random()
        x, f = optimizer(fx, x_start, s_start)

        x_list.append(x)
        f_list.append(f)

    x_avg, x_std = stat(x_list)
    return x_avg, x_std
    # f_avg, f_std = stat(f_list)
    #
    # print("\nx_min:{0:.4f}+{1:.4f}".format(x_avg, x_std))
    # print("f_min:{0:.4f}+{1:.4f}".format(f_avg, f_std))


epsl_r = 1.49E-8
list_one = np.logspace(-2, -6, 5)
list_two = np.logspace(-4, -8, 5)
x_avg_list = []
x_std_list = []
for i in range(0, 4):
    epsl_x = list_one[i]
    epsl_f = list_two[2]
    avg, std = multi_test(fx_one)
    x_avg_list.append(avg)
    x_std_list.append(std)
    plt.errorbar(x=list_one[i], y=x_avg_list[i], yerr=x_std_list[i], fmt='.-')

# axis = np.linspace(-2, -6, 5)
# plt.errorbar(x=axis[0], y=x_avg_list[0], yerr=x_std_list[0], fmt='.-')
plt.show()
#
# for i in range(0, 4):
#     epsl_x = list_one[2]
#     epsl_f = 1ist_two[i]
#     multi_test(fx_one)