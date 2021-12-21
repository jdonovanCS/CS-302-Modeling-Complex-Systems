import numpy as np
from itertools import product
import pylab
import matplotlib.pyplot as plt

from question3_euler_heun import euler_forward_step, heun_step

class MU():

    def __init__(self, K, beta, initial_point, h, nr_timestep):
        '''
        K (list): list of K values
        beta (list): list of beta values
        initial_point (list): list of (M,K) tuples that shows initial values for M and K
        h (list): list of step sizes
        nr_timestep (list): number of steps to run 
        '''

        self.K = K
        self.beta = beta
        self.initial_point = initial_point
        self.h = h
        self.nr_timestep = nr_timestep

    def run(self):
        '''
        runs the system for the given parameter settings
        prints necessary time series plots
        '''
        # for each parameter setting
        for K, beta, initial_point, h, nr_timestep in product( self.K, self.beta, self.initial_point, self.h, self.nr_timestep ):
            self.initialize(initial_point)
            for step in range(nr_timestep):
                self.update(K,beta,h)
                self.observe()
            # plot the results for this parameter setting
            t = np.linspace(0,h*nr_timestep,nr_timestep+1)
            
            fig, ax_left = plt.subplots()
            ax_right = ax_left.twinx()
            
            ax_left.plot(self.M, color='red')
            ax_left.tick_params(axis='y', labelcolor='red')
            ax_left.set_ylabel('population', color='red')
            #ax_left.set_ylim(0,top=K)

            ax_left.plot([],[],' ',label='K: '+str(K)+'\n'+
                                        'beta: '+str(beta)+'\n'+  
                                        'M_0: '+str(initial_point[0])+'\n'+  
                                        'R_0: '+str(initial_point[1]) )  
            ax_left.set_title('mouse universe model')
            ax_left.set_xlabel('timestep')
            ax_left.legend()

            ax_right.plot(self.R, color='green')
            ax_right.tick_params(axis='y', labelcolor='green')
            ax_right.set_ylabel('R', color='green')

            #plt.show()
            plt.savefig('q7 K='+str(K)+' beta='+str(beta)+' init='+str(initial_point)+'.png')


    def initialize(self, initial_point ):
        '''
        initial_point (list): shows initial [M,R]
        '''
        # heun
        self.M = [ initial_point[0] ]
        self.R = [ initial_point[1] ]
        self.m = self.M[0]
        self.r = self.R[0]

    def observe(self):
        # heun
        self.M += [self.m]
        self.R += [self.r]

    def update(self, K, beta, h ):
        '''
        K (float): caryying capacity 
        beta (float): behavioral deterioration
        h (float): step size
        '''
        # calculate r_next
        r_dot = -K*beta # constant rate
        r_next = euler_forward_step( self.r, h, r_dot)
        # calculate m_next_h
        m_dot = self.r*self.m*(1. - (self.m/K))
        m_next_e = euler_forward_step( self.m, h, m_dot )
        m_next_e_dot = r_next*m_next_e*(1. - (m_next_e/K))
        m_next_h = heun_step( self.m, h, m_dot, m_next_e_dot )
        # sanity checks
        if m_next_h < 0:
            m_next_h = 0

        self.m = m_next_h 
        self.r = r_next


if __name__ == '__main__':

    system = MU(K=[3840, 100, 1000], beta=[0.0001, 0.001], initial_point=[(8,2.2),(4,2.2),(4,0.2)], h=[0.01], nr_timestep=[15000])

    system.run()
