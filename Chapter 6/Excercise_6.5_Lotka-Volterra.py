from pylab import *

a = b = c = d = 1.
x = y = 0.1
Dt = .01

def initalize():
    global x, y, result_x, result_y, t, timesteps
    x = 0.1
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
    next_x = x + ((a * x) - (b * x * y)) * Dt
    next_y = y + ((-c * y) + (d * x * y)) * Dt
    x = next_x
    y = next_y
    t = t + Dt

initalize()
while t < 50.:
    update()
    observe()

plot(result_x,result_y) # phasespace
# plot(timesteps,result_x,timesteps,result_y) # over time
show()