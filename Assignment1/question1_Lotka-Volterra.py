from pylab import *

a1 = 1.5
a2 = 1.7
k1 = 1.
k2 = 1.
r1 = 0.5
r2 = 0.5
x = 0.01
y = 1.2
Dt = .001

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
    xvalues, yvalues = meshgrid(arange(0,1,0.001), arange(0,1,0.001))
    xdot = ((r1*xvalues) * ((k1 - xvalues - a2*yvalues)/k2))
    ydot = ((r2*yvalues) * ((k2 - yvalues - a1*xvalues)/k1))

    # Add the nullclines to the plot.
    xmult = []
    xline = []
    yline = []

    # Loop to make the nullclines and remove negative points
    for i in range(0,100):
        xmult.append(i/100.)
        xline.append((i/100.) * (k2/(-k2/a1)) + k2)
        yline.append((i/100.) * ((k1/a2)/-k1) + (k1/a2))
        if xline[-1] < 0:
            xline = xline [:-1]
        if yline[-1] < 0:
            yline = yline [:-1]    

    intersect_x = ((k1/a2)-k2) / ((k2/(-k2/a1))-((k1/a2)/-k1))
    if (intersect_x * (k2/(-k2/a1))+k2) / (intersect_x * ((k1/a2)/-k1) + (k1/a2)):
        intersect_y = intersect_x * (k2/(-k2/a1))+k2
    intersect = Circle((intersect_x, intersect_y), 0.03, color='g', fill=False)
    figure()
    ax = gca()
    ax.streamplot(xvalues, yvalues, xdot, ydot)
    ax.plot(xmult[:len(xline)], xline, 'b', xmult[:len(yline)], yline, 'r')
    ax.plot(k1, 0, 'ro')
    ax.plot(0, k2, 'bo') 
    ax.plot(0, (k1/a2), 'ro')
    ax.text(0, k1/a2, 'k1/a2')
    ax.plot((k2/a1), 0, 'bo')
    ax.text(k2/a1,0,'k2/a1')
    ax.plot(intersect_x, intersect_y, 'go')
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
        ax.text(intersect_x,intersect_y,'equilibrium, unstable')
        ax.text(0, k2, 'k2, stable')
        # ax.text(0,0,'equilibrium, unstable')
    if (a1<1 and a2<1):
        ax.text(k1, 0, 'k1, unstable')
        ax.text(intersect_x,intersect_y,'equilibrium, stable')
        ax.text(0, k2, 'k2, unstable')
        # ax.text(0,0,'equilibrium, unstable')

    ax.add_artist(intersect)
    ax.set_xlabel('N1')
    ax.set_ylabel('N2')
    ax.legend()
    show()


def plot_single_point():
    # For testing an individual point in phase space or over time
    # -----------------------------------------------------------
    initalize()
    while t < 500.:
        update()
        observe()

    plot(result_x,result_y) # phasespace
    show()
    # plot(timesteps,result_x,timesteps,result_y) # over time
    # show()
# -------------------------------------------------------------

# plot_single_point()
streamplot_local()

### Follow up information ###
# The equilibrium point found at the intersection of the two nullclines when:
#  a1 > 0 and a2 > 0
# is unstable. As we can see from the diagram, all of the trajectories move
# away from the center intersection point and toward (0,k2) and (k1, 0)
# which are stable points. The equilibrium point at (0,0) is also unstable
# as can be seen by the trajectories moving away from it as well.