import numpy as np


def euler_forward_step(yt, h, dydt):
    '''
    yt (ndarray): value/s at t
    h (float): step size
    dydt (ndarray): slope/s at yt

    returns (ndarray): euler proxy of y at t+h -- yeth
    '''

    return yt + h * dydt

def euler_forward_method(y_0, h, f_dydt, nr_timestep, t_0):
    '''
    y_0 (ndarray): initial value/s of y
    h (float): step size
    f_dydt (function): takes t,y and outputs dydt
    nr_timestep (int): nr of timesteps to apply euler method
    t_0 (float): initial value of t

    returns (ndarray): array of values of y
    '''

    ys = [y_0]
    for step in range(nr_timestep):
        ys.append( euler_forward_step( ys[step], h, f_dydt( t_0 + step*h, ys[step] ) ) )
        
    return ys

def heun_step( yt, h, dydt, dyedt):
    '''
    yt (ndarray): value/s at t
    h (float): step size
    dydt (ndarray): slope/s at yt
    dyedt (ndarray): slope/s at yeth 
    
    returns (float): heun proxy of y at t+h -- yhth
    '''

    return yt + (h/2.) * ( dydt + dyedt )

def heun_method(y_0, h, f_dydt, nr_timestep, t_0):
    '''
    y_0 (float): initial value of y
    h (float): step size
    f_dydt (function): takes t,y and outputs dydt
    nr_timestep (int): nr of timesteps to apply euler method
    t_0 (float): initial value of t

    returns (ndarray): array of values of y
    '''

    ys = [y_0]
    for step in range(nr_timestep):
        dyedt = f_dydt( t_0 + (step+1)*h, euler_forward_step( ys[step], h, f_dydt( t_0 + step*h, ys[step] ) ) )
        ys.append( heun_step( ys[step], h, f_dydt( t_0 + step*h, ys[step] ), dyedt ) )
        
    return ys


if __name__ == '__main__':

    # test with one dimensional input (float) instead of ndarray
    import pylab

    nr_timestep = 21
    t_0 = 0.
    h = 2./nr_timestep
    f_dydt = lambda t,y: y
    y_0 = 1.0

    # euler 
    y_euler = euler_forward_method( y_0, h, f_dydt, nr_timestep, t_0 )
    # heun 
    y_heun = heun_method( y_0, h, f_dydt, nr_timestep, t_0 )

    t = np.linspace(t_0,h*nr_timestep,nr_timestep+1)
    y_true = np.exp(t)
    
    pylab.plot(t, y_euler, '-r', label='euler')
    pylab.plot(t, y_heun, '-g', label='heun')
    pylab.plot(t, y_true, '-b', label='true')
    pylab.legend(loc='upper right') 
    pylab.show()








