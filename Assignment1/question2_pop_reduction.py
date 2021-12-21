from pylab import *
import random
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Process some arguments')
parser.add_argument('-a1', nargs='?', type=float, default=1.2, help='Rate for species 1')
parser.add_argument('-a2', nargs='?', type=float, default=1.7, help='Rate for species 2')
parser.add_argument('-k1', nargs='?', type=float, default=1., help='Carraying capacity for species 1')
parser.add_argument('-k2', nargs='?', type=float, default=1., help='Carrying capacity for species 2')
parser.add_argument('-r1', nargs='?', type=float, default=.5, help='Rate for species 1')
parser.add_argument('-r2', nargs='?', type=float, default=.5, help='Rate for species 2')
parser.add_argument('-x', nargs='?', type=float, default=.4, help='Starting percentage of x1')
parser.add_argument('-y', nargs='?', type=float, default=.4, help='Starting value for species 2')
parser.add_argument('-Dt', nargs='?', type=float, default=.001, help='Timestep interval')
parser.add_argument('-show_nullclines', nargs='?', type=bool, default=True, help='Display nullclines')
parser.add_argument('-show_streamplot', nargs='?', type=bool, default=True, help='Display streamplot')
parser.add_argument('-p', nargs='?', type=float, default=0.8, help='Decrease percentage for populations')
args = parser.parse_args()

a1 = args.a1
a2 = args.a2
k1 = args.k1
k2 = args.k2
r1 = args.r1
r2 = args.r2
x = xstart = args.x
y = ystart = args.y
Dt = args.Dt
p = args.p
ax = gca()

def initalize():
    global x, y, result_x, result_y, t, timesteps
    result_x = [x]
    result_y = [y]
    t = 0.
    timesteps = [t]

def observe():
    global x, y, result_x, result_y, t, timesteps
    result_x.append(x)
    result_y.append(y)
    timesteps.append(t)

def update():
    global x, y, result_x, result_y, t, timesteps
    next_x = x + ((r1*x) * ((k1 - x - a2*y)/k2) * Dt)
    next_y = y + ((r2*y) * ((k2 - y - a1*x)/k1) * Dt)
    x = next_x
    y = next_y
    t = t + Dt

def streamplot_local():
    # For plotting the phasespace as a streamplot
    # --------------------------------------------------------------
    global ax
    xvalues, yvalues = meshgrid(arange(0,max(k1, k2, k1/a2, k2/a1),0.001), arange(0,max(k1, k2, k1/a2, k2/a1),0.001))
    xdot = ((r1*xvalues) * ((k1 - xvalues - a2*yvalues)/k2))
    ydot = ((r2*yvalues) * ((k2 - yvalues - a1*xvalues)/k1))

    ax.streamplot(xvalues, yvalues, xdot, ydot)


def nullclines():    # Loop to make the nullclines and remove negative points
    global ax
    
    # Add the nullclines to the plot.
    xmult = []
    xline = []
    yline = []

    for i in range(0,100):
        xmult.append(i*max(k1, k2, k1/a2, k2/a1)/100.)
        xline.append((i*max(k1, k2, k1/a2, k2/a1)/100.) * (k2/(-k2/a1)) + k2)
        yline.append((i*max(k1, k2, k1/a2, k2/a1)/100.) * ((k1/a2)/-k1) + (k1/a2))
        if xline[-1] < 0:
            xline = xline [:-1]
        if yline[-1] < 0:
            yline = yline [:-1]    

    intersect_x = ((k1/a2)-k2) / ((k2/(-k2/a1))-((k1/a2)/-k1))
    if (intersect_x * (k2/(-k2/a1))+k2) / (intersect_x * ((k1/a2)/-k1) + (k1/a2)):
        intersect_y = intersect_x * (k2/(-k2/a1))+k2
    
    if not intersect_x > max(k1, k2, k1/a2, k2/a1) and not intersect_y > max(k1, k2, k1/a2, k2/a1):
        intersect = Circle((intersect_x, intersect_y), 0.05, color='g', fill=False)
        ax.plot(intersect_x, intersect_y, 'go')
        if a1 < 1 and a2 < 1:
            ax.text(intersect_x,intersect_y,'equilibrium, stable')
        else:
            ax.text(intersect_x,intersect_y,'equilibrium, unstable')
        ax.add_artist(intersect)

    ax.plot(xmult[:len(xline)], xline, 'b', xmult[:len(yline)], yline, 'r')
    ax.plot(k1, 0, 'ro')
    ax.plot(0, k2, 'bo') 
    ax.plot(0, (k1/a2), 'ro')
    ax.text(0, k1/a2, 'k1/a2')
    ax.plot((k2/a1), 0, 'bo')
    ax.text(k2/a1,0,'k2/a1')
    # ax.plot(0,0,'go')
    if (a1>1 and a2>1):
        ax.text(k1, 0, 'k1, stable')
        ax.text(intersect_x,intersect_y,'equilibrium, unstable')
        ax.text(0, k2, 'k2, stable')
        # ax.text(0,0,'equilibrium, unstable')
    if (a1>1 and a2<1):
        ax.text(k1, 0, 'k1, stable')
        ax.text(intersect_x,intersect_y,'equilibrium, unstable')
        ax.text(0, k2, 'k2, unstable')
        # ax.text(0,0,'equilibrium, unstable')
    if (a1<1 and a2>1):
        ax.text(k1, 0, 'k1, unstable')
        ax.text(0, k2, 'k2, stable')
        # ax.text(0,0,'equilibrium, unstable')
    if (a1<1 and a2<1):
        ax.text(k1, 0, 'k1, unstable')
        ax.text(0, k2, 'k2, unstable')
        # ax.text(0,0,'equilibrium, unstable')
    
    ax.set_xlabel('N1')
    ax.set_ylabel('N2')
    ax.legend()

def plot_single_point():
    # For testing an individual point in phase space or over time
    # -----------------------------------------------------------
    global ax, xstart, ystart
    initalize()
    if ((a2 > 1 and a1 < 1 and y > x*(k2/(-k2/a1)) + k2) or 
        (a1 > 1 and a2 < 1 and y > x*((k1/a2)/-k1) + (k1/a2)) or
        (a1 > 1 and a2 > 1 and a2 > a1 and y > x*(k2/(-k2/a1)) + k2) or
        (a1 > 1 and a2 > 1 and a1 > a2 and y > x*((k1/a2)/-k1) + (k1/a2))):
        if p == 0:
            decrease_populations(random.randint(0,999/1000.))
        else:
            decrease_populations(p)
    while t < 5000.:
        update()
        if ((a2 > 1 and a1 < 1 and y > x*((k1/a2)/-k1) + (k1/a2)) or 
            (a1 > 1 and a2 < 1 and y > x*(k2/(-k2/a1)) + k2) or
            (a1 > 1 and a2 > 1 and a2 > a1 and y > x*((k1/a2)/-k1) + (k1/a2)) or
            (a1 > 1 and a2 > 1 and a1 > a2 and y > x*(k2/(-k2/a1)) + k2)):
            currentx = x
            currenty = y
            if p == 0:
                decrease_populations(random.randint(0,999/1000.))
            else:
                decrease_populations(p)
            
            # decrease_populations(np.random.uniform(min(xstart/currentx, ystart/currenty), max(xstart/currentx,ystart/currenty)))
        observe()
        # if y[0] above k1 line, apply reduction
        # if y value moves above k2 line, apply reduction

    if (ax):
        ax.plot(result_x[0], result_y[0], 'mo')
        ax.text(result_x[0], result_y[0], 'start')
        ax.plot(result_x[-1], result_y[-1], 'mo')
        ax.plot(result_x,result_y, 'm') # phasespace
    else:
        plot(result_x, result_y)
    # plot(timesteps,result_x,timesteps,result_y) # over time
    # show()
# -------------------------------------------------------------

def decrease_populations(p):
    global x, y, xstart, ystart
    # print(p)
    # print(p*x)
    # print(p*y)
    x = xstart = p*x
    y = ystart = p*y



streamplot_local()
nullclines()
plot_single_point()
show()

# ax = gca()
# a1 = .7
# a2 = 1.2
# x = .4
# y=.4
# streamplot_local()
# nullclines()
# plot_single_point()
# show()

# figure()
# ax = gca()
# a1=1.2
# a2=.7
# x=.4
# y=.4
# streamplot_local()
# nullclines()
# plot_single_point()
# show()

# figure()
# ax = gca()
# a1=1.2
# a2=1.7
# x=.4
# y=.4
# streamplot_local()
# nullclines()
# plot_single_point()
# show()

### Follow up information ###
# The observation that organism 2 quickly dominates and reduces organism 1
# to local extinction is a function of the a1 and a2 values (rates) that 
# organism 1 and organism 2 reproduce. If a2 > 1 and a1 < 1, then it will
# not matter if we change the number of organism 2 since their rate is going
# to result in the system again eliminating organism 1. If, however, the
# rates of organism 1 and organism 2 are both above 1, then there will be a
# dependence on the number of organism 1 compared to organism 2, but 
# eventually, one of the organisms in the population will be reduced to 0
# since the stable states of that system force it.

# Decreasing both populations on a timeline where neither population causes 
# the other to go extinct would end up artificially stabilizing the community
# as long as the reductions continue. May need to decrease at a faster rate
# than the birth rates for species. May be tough to show mathematically.

# NEED TO MAKE LOCAL GIT REPO FOR MOCS