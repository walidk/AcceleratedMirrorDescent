__author__ = 'Walid'
import amd.Projections as proj
import numpy as np

class AcceleratedMethod(object):
    def __init__(self, f, gradf, p1, p2, s1, s2, r, x0, name, alpha = 2):
    # f: objective function
    # gradf: gradient of the objective function
    # p1 and p2 are projections
    # p1.project(x, g) returns
    #   in the unconstrained case: nablaPsi(nablaPsiInverse(x) - g)
    #   in the simplex constrained case: \phi(\phi^{-1}(x_i) - g_i + \nu)_+ for an optimal \nu
    # s1, s2 are the step sizes for the primal and dual step
    # r is a scalar parameter in the ODE (needs to be strictly larger than 2)
        self.k = 1
        # projection operators
        self.p1 = p1
        self.p2 = p2
        # step sizes
        self.s1 = s1
        self.s2 = s2
        # Energy rate and averaging rate
        if(r < alpha):
            raise Exception('r needs to be larger than alpha (default 2)')
        self.alpha = alpha
        self.r = r
        # Primal variable initialization
        self.x0 = x0
        self.z0 = x0
        self.xtilde = x0
        self.x = x0
        self.z = x0
        self.xprev = x0
        self.zprev = x0
        # objective and gradient oracles
        self.gradf = gradf
        self.f = f
        self.name = name

    def __str__(self):
        return 'Accelerated method with s1={}, s2={}, p1={}, p2={}, r={}'.format(s1, s2, p1, p2, r)
    def restart(self):
        # print('restarted at iteration {}'.format(self.k))
        self.z = self.xprev
        self.x = self.xprev
        self.k = 1
    def reset(self):
        self.x = self.x0
        self.z = self.x0
        self.xtilde = self.x0
        self.xprev = self.x0
        self.zprev = self.x0
        self.k = 1
    def step(self):
        r = self.r
        k = self.k
        x = self.x
        z = self.z
        g = self.gradf(x)
        xtilde = self.p1.project(x, self.s1*g)
        # xtilde = x
        ztilde = self.p2.project(z, k**(self.alpha - 1)*self.s2/r*g)
        xp = (xtilde + r/k**(self.alpha - 1)*ztilde)/(1+r/k**(self.alpha - 1))
        self.xprev = self.x
        self.zprev = self.z
        self.xtilde = xtilde
        self.x = xp
        self.k = k+1
        self.z = ztilde


class MDMethod(object):
    def __init__(self, f, gradf, p1, s1, x0, name):
    # f: objective function
    # gradf: gradient of the objective function
        self.p1 = p1
        self.s1 = s1
        self.k = 1
        self.x0 = x0
        self.x = x0
        self.xtilde = x0
        self.xprev = x0
        self.gradf = gradf
        self.f = f
        self.name = name

    def __str__(self):
        return 'Accelerated method with s1={}, s2={}, p1={}, p2={}, r={}'.format(s1, s2, p1, p2, r)
    def reset(self):
        self.x = self.x0
        self.xprev = self.x0
        self.k = 1
    def step(self):
        k = self.k
        x = self.x
        g = self.gradf(x)
        xtilde = self.p1.project(x, self.s1*g)
        # xtilde = x
        self.xprev = self.x
        self.xtilde = xtilde
        self.x = xtilde
        self.k = k+1

class AcceleratedMethodWithRestartFunctionScheme(AcceleratedMethod):
    def __init__(self, *args):
        super().__init__(*args)
    # override the step method
    def step(self):
        super().step()
        if(self.f(self.x) > self.f(self.xprev) and self.k > 20):
            # need to restart
            self.restart()


class AcceleratedMethodWithRestartGradScheme(AcceleratedMethod):
    def __init__(self, *args):
        super().__init__(*args)

    # override the step method
    def step(self):
        super().step()
        g = self.gradf(self.xprev)
        if(np.dot(g.T, self.x - self.xprev)>0 and self.k > 20):
            # need to restart
            self.restart()

class AcceleratedMethodWithSpeedRestart(AcceleratedMethod):
    def __init__(self, *args):
        self.crit = 0
        super().__init__(*args)

    # override the step method
    def step(self):
        super().step()
        crit = self.r/self.k*(self.z - self.x)
        if(np.linalg.norm(crit) < np.linalg.norm(self.crit) and self.k > 20):
            # need to restart
            self.crit = 0
            self.restart()
        else:
            self.crit = crit

class UnconstrainedAcceleratedMethod(AcceleratedMethod):
    def __init__(self, f, gradf, s1, s2, r, x0, name):
        p = proj.NoProjection()
        super().__init__(f, gradf, p, p, s1, s2, r, x0, name)

# def descent(s, gradPsi, gradf):
#     # s is the discretization step size
#     # gradPsi: Bregman projection function
#     # gradf: gradient of the objective function
#     def step(k, x, z):
#         xp = gradPsi(z)
#         zp = z - s*gradf(xp)
#         return (xp, zp)
#     return step


