__author__ = 'Walid'

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

nan = np.NaN

# f: objective function
# grad: gradient
# Lyap: Lyapunov function. Lyap(x, z, k)
# Breg: Bregman divergence. Breg(xstar, x)
# methods: list of optimization methods.
# horizon: time horizon
# video_length: in seconds
# video_step: in frames (interval between two frames)
# ymin: minimal objective value to be plotted (to avoid numerical inaccuracy)
# ymax: minimal objective value to be plotted
# min_dist: minimal distance to the minimizer to be plotted (to avoid numerical inaccuracy)
def simulateSimplex(f, grad, Lyap, Breg, methods, horizon, video_length, video_step, title, ymin, ymax, view3d, homothetic_center, filterSmallValue):

    # ==================================================================================================================
    # Initialize
    # ==================================================================================================================
    colors = ['b', 'g', 'r', 'c', 'm']
    vertices = np.array([[1, 0], [np.cos(2*np.pi/3), np.sin(2*np.pi/3)], [np.cos(4*np.pi/3), np.sin(4*np.pi/3)]]).T
    delta_coeff = 1
    delta_coeff3 = .2
    simplex_coeff = 15
    simplex_mesh_size = 40
    zoom_interval = 50
    times = [0, 20, 25, 100, 200, 300, horizon - zoom_interval]

    # compute the limit point, take fstar to be the value there
    fstar = np.Infinity

    for method in methods:
        for t in range(horizon+100):
            method.step()
            if(f(method.x) < fstar):
                fstar = f(method.x)
                xstarMat = method.x
        method.reset() # make sure the method is reset before continuing

    xstar = np.hstack(xstarMat)
    print(xstar)
    print(fstar)
    def fMinusFStar(x):
        return f(x) - fstar

    def toSimplex(x):
        return np.dot(vertices, x-xstar)

    simplexPoints = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]).T
    simplexBoundary = np.zeros((2, 4))
    for i in range(4):
        simplexBoundary[:, i] = toSimplex(simplexPoints[:, i])

    #  generate simplex mesh points
    def simplexMesh(n):
        x1 = 1; y1 = 0
        x2= -.5; y2 = np.sqrt(3)/2
        x3 = -.5; y3 = -np.sqrt(3)/2
        points = np.zeros((3, n*(n+1)/2))
        idx = 0
        for i in range(n):
            for j in range(n-i):
                x = -1/2 + 3/2*(i/n)
                y = np.sqrt(3)/2*(1-i/n) - j*np.sqrt(3)/n
                p1 = max(((y2-y3)*(x-x3)+(x3-x2)*(y-y3))/((y2-y3)*(x1-x3)+(x3-x2)*(y1-y3)), 0)
                p2 = max(((y3-y1)*(x-x3)+(x1-x3)*(y-y3))/((y2-y3)*(x1-x3)+(x3-x2)*(y1-y3)), 0)
                p3 = max(1-p1-p2, 0)
                p = np.array([p1,p2,p3]).T
                points[:, idx] = p
                idx += 1
        return points

    meshPoints = simplexMesh(simplex_mesh_size)

    def geoMean(x):
        return np.exp(np.mean(np.log(x)))

    # a function that will plot a list of points with fading color
    def fading_plot(ax, points, color, linewidth, dashes):
        Imax = 10
        M2 = np.size(points, 1)
        for i in range(Imax):
            M1 = max(0, M2-20)
            if(i == Imax-1):
                M1 = 0
            line, = ax.plot(points[0, M1:M2], points[1, M1:M2], color, alpha = 1 - .5*i/Imax, linewidth = linewidth)
            line.set_dashes(dashes)
            M2 = M1+1
        return ax

    def fading_plot3(ax, points, values, color, linewidth, linestyle):
        Imax = 10
        M2 = np.size(points, 1)
        for i in range(Imax):
            M1 = max(0, M2-20)
            if(i == Imax-1):
                M1 = 0
            ax.plot(points[0, M1:M2], points[1, M1:M2], zs = values[M1:M2], color = color, alpha = 1 - .5*i/Imax, linewidth = linewidth, ls=linestyle)
            # line.set_dashes(dashes)
            M2 = M1+1
        return ax

    def setAxisZoom(ax, points, combineDeltas, delta, delta_coeff):
        # take a box around the the last few values
        deltaXs = [np.max(np.abs(points[m][0,:])) for m in ms[0:]]
        deltaYs = [np.max(np.abs(points[m][1,:])) for m in ms[0:]]
        delta[1] = min(delta[1], delta_coeff*max(combineDeltas(deltaXs), combineDeltas(deltaYs)))
        lim = delta[1]

        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        # ax.locator_params(tight=True, nbins=5)

    def setAxisZoom3(ax, points, combineDeltas, delta, delta_coeff):
        # take a box around the the last few values
        deltaXs = [np.max(np.abs(points[m][0,:])) for m in ms]
        deltaYs = [np.max(np.abs(points[m][1,:])) for m in ms]
        delta[1] = min(delta[1], delta_coeff*max(combineDeltas(deltaXs), combineDeltas(deltaYs)))
        lim = delta[1]
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_zlim([0, lim*ymax])
        ax.view_init(view3d[0], view3d[1])

    def drawSurface(ax, h):
        hh = np.power(2, np.ceil(np.log2(min(h, 1))))
        center = np.hstack(homothetic_center)
        n = np.size(meshPoints, 1)
        coords = np.zeros((2, n))
        vals = np.zeros((1, n))
        for i in range(n):
            p = meshPoints[:, i]
            p = center + hh*(p-center)
            coords[:, i] = toSimplex(p)
            vals[0, i] = fMinusFStar(p)
            # if(vals[0, i] > h*ymax):
            #     vals[0,i] = nan
            # ignore values that are too large
        ax.plot_trisurf(coords[0, :], coords[1, :], vals[0, :], cmap=cm.Blues, linewidth=0.2, alpha=.6)


    # ==================================================================================================================
    # Initialization
    # ==================================================================================================================
    nb_methods = len(methods)
    ms = range(nb_methods)
    values = {} # objective value
    valuesZ = {} # objective value
    xs = {} # primal variable
    zs = {} # projected dual variable
    Vs = {} # Lyapunov function values
    divs = {} # Bregman divergence
    delta = {0: {1: 1},
             1: {1: 1},
             2: {1: 1},
             3: {1: 1}}

    for m in ms:
        values[m] = np.zeros((horizon, 1))
        valuesZ[m] = np.zeros((horizon, 1))
        Vs[m] = np.zeros((horizon, 1))
        divs[m] = np.zeros((horizon, 1))
        xs[m] = np.zeros((2, horizon))
        zs[m] = np.zeros((2, horizon))

    # ==================================================================================================================
    # Run the algorithms
    # ==================================================================================================================
    for m in ms:
        method = methods[m]
        r = method.r if hasattr(method, 'r') else 3
        for k in range(horizon):
            values[m][k] = fMinusFStar(method.x)
            xMat = method.x
            zMat = method.z if hasattr(method, 'z') else method.x
            valuesZ[m][k] = fMinusFStar(zMat)
            x = np.hstack(xMat)
            z = np.hstack(zMat)
            xs[m][:, k] = toSimplex(x)
            zs[m][:, k] = toSimplex(z)
            Vs[m][k] = Lyap(xMat, zMat, xstarMat, k, r)
            divs[m][k] = Breg(xstarMat, zMat)
            # filter out values that are too small (to avoid numerical inaccuracy)
            if(filterSmallValue and values[m][k] < ymin):
                values[m][k] = nan
                valuesZ[m][k] = nan
                Vs[m][k] = nan
                divs[m][k] = nan
            method.step()

    # ==================================================================================================================
    # Plot
    # ==================================================================================================================
    figsize = (12, 8)
    fig = plt.figure(figsize=figsize)
    plt.show()

    global min_value, max_value, min_Lyap, max_Lyap
    min_value = max(ymin, min([np.nanmin(values[m]) for m in ms]))
    max_value = max([np.nanmax(values[m]) for m in ms])
    min_Lyap = max(ymin, min([np.nanmin(Vs[m]) for m in ms]))/50
    max_Lyap = 2*max([np.nanmax(Vs[m]) for m in ms])

    def plotFrom(n1):
        n2 = n1+zoom_interval
        fig = plt.figure(figsize=figsize)
        # top plot contains function values in log log scale
        ax = fig.add_subplot(221)
        for m in ms:
            ax.plot(values[m], label=methods[m].name)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend(loc=3)
        ax.set_xlim([1, horizon])
        ax.set_ylim([min_value, max_value])
        ax.set_xlabel('k')
        ax.set_ylabel('f(x(k))')

        # Lyapunov energy values in log log scale
        ax = fig.add_subplot(222)
        for m in ms:
            ax.plot(Vs[m], label=methods[m].name)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend(loc=3)
        ax.set_xlim([1, horizon])
        ax.set_ylim([min_Lyap, max_Lyap])
        ax.set_xlabel('k')
        ax.set_ylabel('Lyap(x(k), z(k), k)')

        # Plot 3 and 4: plot a subset of the trajectory
        points = [xs[m][:, n1:n2] for m in ms]
        vals = [values[m][n1:n2] for m in ms]

        # trajectory
        ax = fig.add_subplot(223)
        for m in ms:
            ax.plot(xs[m][0,:], xs[m][1,:], colors[m])
            ax.plot(zs[m][0,:], zs[m][1,:], colors[m]+'--')
        setAxisZoom(ax, points, np.mean, delta[0], delta_coeff)
        ax.set_xlabel('x0(k)')
        ax.set_ylabel('x1(k)')
        # 3D trajectory
        ax = fig.add_subplot(224, projection='3d')
        ax.axis('off')
        # ax.plot(simplexBoundary[0, :], simplexBoundary[1, :], zs = 0, color='k', alpha=.4)
        setAxisZoom3(ax, points, np.mean, delta[1], delta_coeff3)
        drawSurface(ax, simplex_coeff*delta[1][1])
        for m in ms:
            ax.plot(xs[m][0,0:n2], xs[m][1,0:n2], zs = np.hstack(values[m][0:n2]), color=colors[m])

        # plot function values and trajectory
        # drawSurface(ax, delta2[1])
        # for m in ms:
        #     ax.plot(xs[m][0,:], xs[m][1,:], zs = np.hstack(values[m]), color=colors[m])
        #     ax.plot(xs[m][0,:], xs[m][1,:], zs = 0, color=colors[m], ls='--')


        # Divergences in the dual space
        # ax = fig.add_subplot(224)
        # for m in ms:
        #     ax.plot(divs[m], label=methods[m].name)
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        # ax.set_xlabel('k')
        # ax.set_ylabel('Breg(xstar, x(k))')
        # ax.legend(loc=3)
        # ax.set_xlim([1, horizon])
        plt.show()

    # ==================================================================================================================
    # Create and export video
    # ==================================================================================================================

    # fig = plt.figure(figsize=figsize)

    def update_plot(frameNb):
        sys.stdout.write('{}%\r'.format(int(100*(frameNb+1)*video_step/horizon)))
        n = video_step*frameNb + 5

        #  Figure 1: objective
        ax = fig.add_subplot(221)
        ax.clear()
        for m in ms:
            ax.plot(values[m][0:n, :], label=methods[m].name)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend(loc=3)
        ax.set_xlim([1, horizon])
        ax.set_ylim([min_value, max_value])
        ax.set_title('Objective function')

        #  Figure 2: Lyapunov function
        ax = fig.add_subplot(222)
        ax.clear()
        for m in ms:
            ax.plot(Vs[m][0:n, :], label=methods[m].name)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend(loc=3)
        ax.set_xlim([1, horizon])
        ax.set_ylim([min_Lyap, max_Lyap])
        ax.set_title('Lyapunov function')
        n1 = max(n-zoom_interval, 0)
        n2 = min(n, horizon)
        points = [xs[m][:, n1:n2] for m in ms]

        # Figure 3
        ax = fig.add_subplot(223)
        ax.clear()
        for m in ms:
            ax = fading_plot(ax, xs[m][:, 0:n], colors[m], 1, [])
            ax = fading_plot(ax, zs[m][:, 0:n], colors[m], .5, [2, 1])
        setAxisZoom(ax, points, np.mean, delta[2], delta_coeff)
        ax.set_title('Trajectory')

        # Figure 4
        ax = fig.add_subplot(224, projection='3d')
        ax.axis('off')
        for m in ms:
            ax = fading_plot3(ax, xs[m][:, 0:n], np.hstack(values[m]), colors[m], 1, '-')
        setAxisZoom3(ax, points, np.mean, delta[3], delta_coeff3)
        drawSurface(ax, simplex_coeff*delta[3][1])
        # setAxisZoom3(ax, points, vals, np.min, delta2, 2)
        # ax = fig.add_subplot(224)
        # ax.clear()
        # for m in ms:
        #     ax = fading_plot(ax, xs[m][:, 0:n], colors[m], 1, [])
        #     ax = fading_plot(ax, zs[m][:, 0:n], colors[m], .5, [2, 1])
        # setAxisZoom(ax, points, np.min, delta2, 2)
        # ax.set_title('Trajectory')

        return ax

    for t in times:
        plotFrom(t)


    # Get input from the user
    export_data = 0
    export_data = int(input("Export data? (0: no, 1: yes)"))

    # Creating the Animation object
    if(export_data):
        i = 0
        # make sure we are not overwriting an existing directory
        while(os.path.exists('out/{}_{}/'.format(title, i))):
            i+=1
        dir = 'out/{}_{}/'.format(title, i)
        os.makedirs(dir)
        print('exporting data to' + dir)
        saveLinear = 1
        saveCSV = 1
        # Export data to CSV:
        path = dir+'data.csv'
        file = open(path, 'w')
        header = "k, "
        for m in ms:
            header+="f_{}, L_{}, x0_{}, x1_{}, z0_{}, z1_{}, ".format(m, m, m, m, m, m)
        file.write(header+"\n")
        for k in range(horizon):
            line = "{}, ".format(k)
            for m in ms:
                line+="{}, {}, {}, {}, {}, {}, ".format(values[m][k, 0], Vs[m][k, 0], xs[m][0, k], xs[m][1, k], zs[m][0, k], zs[m][1, k])
            file.write(line+"\n")
        # Export video
        path = dir+'traj_linear.mp4'
        ani = animation.FuncAnimation(fig, update_plot, int(horizon/video_step - 2), interval=video_length/horizon*1000*video_step, blit=False)
        ani.save(path, extra_args=['-vcodec', 'libx264'])


