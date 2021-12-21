from pylab import *

xvalues, yvalues = meshgrid(arange(0,3,0.1), (0,3,0.1))

xdot = xvalues - xvalues * yvalues
ydot = -yvalues + xvalues * yvalues

streamplot(xvalues, yvalues, xdot, ydot)
show()