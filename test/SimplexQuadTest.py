__author__ = 'walid'

import numpy as np
import matplotlib.pyplot as plt
import amd.AcceleratedMethods as ac
import math
import amd.Projections as proj
import amd.SimulationSimplex as sim

# ======================================================================================================================

# Generate the objective function
d = 3

xstar = np.array(
[0.47, 0.13, 0.4]
# [0.44, 0.08, 0.48]
# [0.45, 0.3, 0.25]
# [0.3, 0.05, 0.65]
# [0, .3, .7]
).T
print(xstar)

# xstar = np.random.rand(d, 1)
xstar = xstar/np.sum(xstar)
A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
print(A)
def KLdiv(x, y):
    div = sum(xi*np.log(xi/yi) for xi, yi in zip(x, y) if yi>0 and xi>0)
    return max(0, div)

def f(x):
    return np.dot(np.dot((x-xstar).T, A), x-xstar)/2
    # return KLdiv(x, xstar)
def gradf(x):
    return np.dot(A, x-xstar)
    # return 1 + np.log(x) - np.log(xstar)

def Lyap(x, z, xstar, k, r):
    return s*k*k*(f(x) - f(xstar)) + r*r*KLdiv(xstar, z)


# ======================================================================================================================
# run the descent


precision = 1e-10
epsilon = .3
# Simplex constrained projections
psp = proj.SimplexProjectionPNorm(p = 1.9, precision=precision)
ps2 = proj.SimplexProjectionEuclidean(precision=precision)
ps1 = proj.SimplexProjectionPNorm(p=1.2, precision=precision)
psExp = proj.SimplexProjectionExp(dimension = d, precision=precision, epsilon = 1/d)
psExpS = proj.SimplexProjectionExpSort(dimension = d, epsilon = epsilon)
psExpS0 = proj.SimplexProjectionExpSort(dimension = d, epsilon = 0)
# Unconstrained transformation
noProj = proj.NoProjection()
# pExp = proj.PotentialProjectionExp(dimension = d, epsilon = 1/d)


lmax = 20
s = 1/lmax
r = 3
p1 = psExpS
p2 = psExpS0
s1 = s*p1.epsilon/(1+d*p1.epsilon)
print(s1)
s2 = s

x0 = np.array([.55, .25, .2]).T


amd = ac.AcceleratedMethod(f, gradf, p1, p2, s1, s2, r, x0, 'accelerated descent')
amdrf = ac.AcceleratedMethodWithRestartFunctionScheme(f, gradf, p1, p2, s1, s2, r, x0, 'function restart')
amdrg = ac.AcceleratedMethodWithRestartGradScheme(f, gradf, p1, p2, s1, s2, r, x0, 'gradient restart')
amdrs = ac.AcceleratedMethodWithSpeedRestart(f, gradf, p1, p2, s1, s2, r, x0, 'speed restart')
md = ac.MDMethod(f, gradf, p2, s2, x0, 'mirror descent')

# Test with different value of r
rs = [3, 10, 30]

methods = [md,
           amd,
           # amdrs,
           # amdrg
           ]

methods = [ac.AcceleratedMethod(f, gradf, p1, p2, s1, s2, r, x0, 'r = {}'.format(r)) for r in rs]


# plot and video parameters

# sim.simulateSimplex(
#     f, gradf, Lyap, KLdiv, methods,
#     horizon = 400,
#     video_length = 15,
#     video_step = 2,
#     title = 'AMD_KLdiv_with_{}_{}_r={}'.format(p1, p2, r),
#     ymin = 1e-13,
#     ymax = 50e-1,
#     view3d = [30, -160],
#     homothetic_center = xstar)

# xstar = np.array([[0.47, 0.13, 0.4]])
# sim.simulateSimplex(
#     f, gradf, Lyap, KLdiv, methods,
#     horizon = 400,
#     video_length = 15,
#     video_step = 2,
#     title = 'AMD_KLdiv_with_{}_{}_r={}'.format(p1, p2, r),
#     ymin = 1e-13,
#     ymax = 15e-1,
#     view3d = [25, -100],
#     homothetic_center = xstar)

# xstar = np.array([[0.44, 0.08, 0.48]])
sim.simulateSimplex(
    f, gradf, Lyap, KLdiv, methods,
    horizon = 400,
    video_length = 15,
    video_step = 7,
    title = 'AMD_KLdiv_with_{}_{}_r={}'.format(p1, p2, r),
    ymin = 1e-11,
    ymax = 5e-1,
    view3d = [20, -100],
    homothetic_center = xstar,
    filterSmallValue = False)
