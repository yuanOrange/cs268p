import numpy as np
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS
from scipy.optimize import SR1
from scipy.optimize import minimize
import time

np.random.seed(int(time.time()))

R = 10
r = [1, 1, 1]
m = [1, 1, 1]


def compute_xyz(alpha, i):
    xi = (R - r[i]) * np.cos(alpha[i]) * np.cos(alpha[i + 3])
    yi = (R - r[i]) * np.cos(alpha[i]) * np.sin(alpha[i + 3])
    zi = (R - r[i]) * np.sin(alpha[i])
    return xi, yi, zi


def compute_dist(alpha, i, j):
    xi, yi, zi = compute_xyz(alpha, i)
    xj, yj, zj = compute_xyz(alpha, j)
    return np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)


def obj_func(alpha):
    w = np.zeros(3)
    for i in range(3):
        w[i] = - m[i] * (R - r[i]) * np.cos(alpha[i]) * np.sin(alpha[i + 3])
    return np.sum(w)


def obj_grad(alpha):
    grad = np.zeros_like(alpha)
    for i in range(3):
        grad[i] = m[i] * (R - r[i]) * np.sin(alpha[i + 3]) * np.sin(alpha[i])
        # grad[:3] = m[:] * (R - r[:]) * np.sin(alpha[3:]) * np.sin(alpha[:3])
        grad[i + 3] = - m[i] * (R - r[i]) * np.cos(alpha[i]) * np.cos(alpha[i + 3])
    return grad


def obj_hess(alpha):
    hess = np.zeros((6, 6))
    for i in range(3):
        hess[i][i] = hess[i + 3][i + 3] = m[i] * (R - r[i]) * np.sin(alpha[i + 3]) * np.cos(alpha[i])
        hess[i][i + 3] = hess[i + 3][i] = m[i] * (R - r[i]) * np.sin(alpha[i]) * np.cos(alpha[i + 3])
    return hess


def cons_func(alpha):
    dist_01 = compute_dist(alpha, 0, 1)
    dist_12 = compute_dist(alpha, 1, 2)
    dist_20 = compute_dist(alpha, 2, 0)
    return [dist_01, dist_12, dist_20]


def cons_grad(alpha):
    v = np.zeros(3)
    sin_theta = np.zeros(3)
    cos_theta = np.zeros(3)
    sin_phi = np.zeros(3)
    cos_phi = np.zeros(3)
    grad = np.full((3, 6), 0)

    # v[:] = [R] * 3 - r[:]
    for i in range(3):
        v[i] = R - r[i]
    sin_theta[:] = np.sin(alpha[:3])
    sin_phi[:] = np.sin(alpha[3:])
    cos_theta[:] = np.cos(alpha[:3])
    cos_phi[:] = np.cos(alpha[3:])

    for i in range(3):
        j = (i + 1) % 3
        grad[i][i] = (v[i] * v[j] / cons_func(alpha)[i]) * (cos_phi[i] * cos_theta[j] * cos_phi[j] * sin_theta[i] +
                                                            sin_phi[i] * cos_theta[j] * sin_phi[j] * sin_theta[i] -
                                                            sin_theta[j] * cos_theta[i])

        grad[i][j] = (v[i] * v[j] / cons_func(alpha)[i]) * (cos_theta[i] * cos_phi[j] * cos_phi[j] * sin_theta[j] +
                                                            cos_theta[i] * sin_phi[i] * sin_phi[j] * sin_theta[j] -
                                                            sin_theta[i] * cos_theta[j])

        grad[i][i + 3] = (v[i] * v[j] / cons_func(alpha)[i]) * (cos_theta[i] * cos_theta[j] * cos_phi[j] * sin_phi[i] -
                                                                cos_theta[i] * cos_theta[j] * sin_phi[j] * cos_phi[i])

        grad[i][j + 3] = (v[i] * v[j] / cons_func(alpha)[i]) * (cos_theta[i] * cos_phi[j] * cos_theta[i] * sin_phi[j] -
                                                                cos_theta[i] * sin_phi[i] * cos_theta[j] * cos_phi[j])

    return grad


def cons_hess(phi, v):
    pass
    # hess1 = np.full((6, 6), 0)
    # hess2 = np.full((6, 6), 0)
    # hess3 = np.full((6, 6), 0)
    #
    # for i in range(6):
    #     hess1[i][i] =1
    #
    # return v[0] * hess1 + v[1] * hess2 + v[2] * hess3


lin_cons_matrix = np.eye(6)
lin_cons_lb = [-np.pi/2, -np.pi/2, -np.pi/2, 0, 0, 0]
lin_cons_ub = [np.pi/2, np.pi/2, np.pi/2, 2 * np.pi, 2 * np.pi, 2 * np.pi]
linear_constraint = LinearConstraint(lin_cons_matrix, lin_cons_lb, lin_cons_ub)

nonlin_cons_lb = [r[0] + r[1], r[1] + r[2], r[2] + r[0]]
nonlinear_constraint = NonlinearConstraint(cons_func, nonlin_cons_lb, np.inf, jac=cons_grad, hess='2-point')

phi0 = np.zeros(6)
for i in range(6):
    phi0[i] = np.random.uniform(0, np.pi)
phi0 = [0, np.pi/4, -np.pi/4, 0, 0, 0]
print('phi0: ', phi0)
res = minimize(obj_func, phi0, method='trust-constr', jac=obj_grad, hess=obj_hess,
               constraints=[linear_constraint, nonlinear_constraint], options={'verbose': 1})
print(res.x)

v0 = compute_xyz(res.x, 0)
v1 = compute_xyz(res.x, 1)
v2 = compute_xyz(res.x, 2)
print('v0: ', v0)
print('v1: ', v1)
print('v2: ', v2)