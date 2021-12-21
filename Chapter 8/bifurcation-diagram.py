from pylab import *

def xeq1(r):
    return sqrt(r)

def xeq2(r):
    return -sqrt(r)

domain = linspace(0,100)
plot(domain, xeq1(domain), 'b-', linewidth=3)
plot(domain, xeq2(domain), 'r--', linewidth=3)
plot([0],[0],'go')
axis([-10, 100, -10, 10])
xlabel('r')
ylabel('x_eq')

show()