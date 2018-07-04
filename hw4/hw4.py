import numpy as np
from scipy.optimize import minimize
import time

np.random.seed(int(time.time()))


def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0) + 1


def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200 * (xm - xm_m1 ** 2) - 400 * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm)
    der[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2] ** 2)
    return der


def rosen_hess(x):
    x = np.asarray(x)
    H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200 * x[0] ** 2 - 400 * x[1] + 2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200 * x[1:-1] ** 2 - 400 * x[2:]
    H = H + np.diag(diagonal)
    return H


def rosen_hess_p(x, p):
    x = np.asarray(x)
    Hp = np.zeros_like(x)
    Hp[0] = (1200 * x[0] ** 2 - 400 * x[1] + 2) * p[0] - 400 * x[0] * p[1]
    Hp[1:-1] = -400 * x[:-2] * p[:-2] + (202 + 1200 * x[1:-1] ** 2 - 400 * x[2:]) * p[1:-1] \
               - 400 * x[1:-1] * p[2:]
    Hp[-1] = -400 * x[-2] * p[-2] + 200 * p[-1]
    return Hp


def test(x):
    x = np.asarray(x)
    return (x[0] - 1) ** 2 + (x[0] + 1) ** 2 * sum((x[1:] - 1) ** 2) + 1


def test_der(x):
    x = np.asarray(x)
    der = np.zeros_like(x)
    der[0] = 2 * (x[0] - 1) + 2 * (x[0] + 1) * sum((x[1:] - 1) ** 2)
    der[1:] = 2 * (x[0] + 1) ** 2 * (x[1:] - 1)
    return der


def test_hess(x):
    x = np.asarray(x)
    H = np.diag([0] * len(x), 0)
    H[0][0] = 2 + 2 * sum((x[1:] - 1) ** 2)
    for i in range(1, len(x)):
        H[i][i] = 2 * (x[0] + 1) ** 2
        H[i][0] = H[0][i] = 2 * (x[i] - 1) * 2 * (x[0] + 1)
    return H


def print_res(res):
    print('\tx_test: ', res.x)
    print('\tx_rel_err: ', np.linalg.norm(res.x - x_real) / np.linalg.norm(x_real))
    print('\tf_test: ', rosen(res.x))
    print('\tf_rel_err: ', (rosen(res.x) - f_real) / f_real)


dim = 10
x0 = np.zeros(dim)
for i in range(dim):
    x0[i] = np.random.uniform(-3, 3)
print('x0: ', x0)
x_real = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
f_real = 1

# rosen_brock function

res = minimize(rosen, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': False})
print('Nelder-Mead:')
print_res(res)

res = minimize(rosen, x0, method='BFGS', jac=rosen_der, options={'disp': False})
print('BFGS: ')
print_res(res)

res = minimize(rosen, x0, method='Newton-CG', jac=rosen_der, hess=rosen_hess, options={'xtol': 1e-8, 'disp': False})
print('Newton-CG: ')
print_res(res)

res = minimize(rosen, x0, method='Newton-CG', jac=rosen_der, hessp=rosen_hess_p, options={'xtol': 1e-8, 'disp': False})
print('Newton-CG 2: ')
print_res(res)

res = minimize(rosen, x0, method='trust-ncg', jac=rosen_der, hess=rosen_hess, options={'gtol': 1e-8, 'disp': False})
print('trust-ncg: ')
print_res(res)

res = minimize(rosen, x0, method='trust-ncg', jac=rosen_der, hessp=rosen_hess_p, options={'gtol': 1e-8, 'disp': False})
print('trust-ncg 2: ')
print_res(res)

res = minimize(rosen, x0, method='trust-krylov', jac=rosen_der, hess=rosen_hess, options={'gtol': 1e-8, 'disp': False})
print('trust-krylov: ')
print_res(res)

res = minimize(rosen, x0, method='trust-krylov', jac=rosen_der, hessp=rosen_hess_p,
               options={'gtol': 1e-8, 'disp': False})
print('trust-krylov 2: ')
print_res(res)

res = minimize(rosen, x0, method='trust-exact', jac=rosen_der, hess=rosen_hess, options={'gtol': 1e-8, 'disp': False})
print('trust-exact: ')
print_res(res)

# test function

res = minimize(test, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': False})
print('Nelder-Mead:')
print_res(res)

res = minimize(test, x0, method='BFGS', jac=test_der, options={'disp': False})
print('BFGS: ')
print_res(res)

res = minimize(test, x0, method='Newton-CG', jac=test_der, hess=test_hess, options={'xtol': 1e-8, 'disp': False})
print('Newton-CG: ')
print_res(res)

res = minimize(test, x0, method='trust-ncg', jac=test_der, hess=test_hess, options={'gtol': 1e-8, 'disp': False})
print('trust-ncg: ')
print_res(res)

res = minimize(test, x0, method='trust-krylov', jac=test_der, hess=test_hess, options={'gtol': 1e-8, 'disp': False})
print('trust-krylov: ')
print_res(res)

res = minimize(test, x0, method='trust-exact', jac=test_der, hess=test_hess, options={'gtol': 1e-8, 'disp': False})
print('trust-exact: ')
print_res(res)
