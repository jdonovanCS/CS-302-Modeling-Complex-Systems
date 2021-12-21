import numpy as np
from itertools import product
import pylab

from question3_euler_heun import euler_forward_step, heun_step

class SIS():

    def __init__(self, alpha, beta, population, h, nr_timestep):
        '''
        alpha (list): list of alpha values
        beta (list): list of beta values
        population (list): list of (S,I) tuples that shows initial values for S and I
        h (list): list of step sizes
        nr_timestep (list): number of steps to run 
        '''

        self.alpha = alpha
        self.beta = beta
        self.population = population
        self.h = h
        self.nr_timestep = nr_timestep

    def run(self):
        '''
        runs the system for the given parameter settings
        prints necessary time series plots
        '''
        # for each parameter setting
        for alpha, beta, population, h, nr_timestep in product( self.alpha, self.beta, self.population, self.h, self.nr_timestep ):
            self.initialize(population)
            for step in range(nr_timestep):
                self.update(alpha,beta,h)
                self.observe()
            # plot the results for this parameter setting
            t = np.linspace(0,h*nr_timestep,nr_timestep+1)
            
            pylab.plot(t, self.S_e, 'g-', label='euler S')
            pylab.plot(t, self.I_e, 'r-', label='euler I')
            pylab.plot(t, self.S_h, 'g--', label='heun S')
            pylab.plot(t, self.I_h, 'r--', label='heun I')
            pylab.ylim(0,100)
            pylab.xlabel('t')
            pylab.ylabel('population')
            pylab.title('h='+str(h)+' beta='+str(beta))
            pylab.legend(loc='upper right') 
            pylab.savefig('q4 h='+str(h)+' beta='+str(beta)+'.png')
            pylab.close()
            #pylab.show()


    def initialize(self, population ):
        '''
        population (list): shows initial population [S,I]
        '''
        # euler
        self.S_e = [ population[0] ]
        self.I_e = [ population[1] ]
        self.s_e = self.S_e[0]
        self.i_e = self.I_e[0]
        # heun
        self.S_h = [ population[0] ]
        self.I_h = [ population[1] ]
        self.s_h = self.S_h[0]
        self.i_h = self.I_h[0]

        self.N = population[0] + population[1]# constant

    def observe(self):
        # euler
        self.S_e += [self.s_e]
        self.I_e += [self.i_e]
        # heun
        self.S_h += [self.s_h]
        self.I_h += [self.i_h]

    def update(self, alpha, beta, h ):
        '''
        alpha (float): parameter 
        beta (float): parameter
        h (float): step size
        '''
        # calculate s_next_e
        s_dot = - beta*self.s_e*(self.N-self.s_e) + alpha*(self.N-self.s_e)
        s_next_e = euler_forward_step( self.s_e, h, s_dot )
        # calculate s_next_h
        s_dot_ = - beta*self.s_h*(self.N-self.s_h) + alpha*(self.N-self.s_h)
        s_next_e_ = euler_forward_step( self.s_h, h, s_dot_ )
        s_next_e_dot = - beta*s_next_e_*(self.N-s_next_e_) + alpha*(self.N-s_next_e_)
        s_next_h = heun_step( self.s_h, h, s_dot_, s_next_e_dot )
        # sanity checks euler
        if s_next_e > self.N:
            s_next_e = self.N
        elif s_next_e < 0:
            s_next_e = 0
        # sanity checks heun
        if s_next_h > self.N:
            s_next_h = self.N
        elif s_next_h < 0:
            s_next_h = 0
        # calculate i_next
        i_next_e = self.N - s_next_e
        i_next_h = self.N - s_next_h

        self.s_e = s_next_e 
        self.i_e = i_next_e
        self.s_h = s_next_h 
        self.i_h = i_next_h


if __name__ == '__main__':

    system = SIS(alpha=[0.25], beta=[0.03,0.06,0.1], population=[(90,10)], h=[0.01,0.5,2.0], nr_timestep=[50])

    system.run()
