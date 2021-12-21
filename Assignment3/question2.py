import numpy as np
import pylab


def raw_pk(k):
    return 0.25*(0.75**k)

def get_pk_dist(k_max):
    pk_dist = [raw_pk(k+1) for k in range(k_max)]
    pk_dist = np.asarray(pk_dist)
    return pk_dist / np.sum(pk_dist)

class CONFIGURATION_MODEL():

    def __init__(self, alpha, beta, population, h, nr_timestep, k_max, vaccination_rate, vaccination_effect, vaccination):
        '''
        alpha : recovery rate
        beta : transmission rate
        population : full list of I_k s
        h : step size
        nr_timestep : number of steps to run 
        k_max : maximum nr of edge 
        '''

        self.alpha = alpha
        self.beta = beta
        self.population = population
        self.h = h
        self.nr_timestep = nr_timestep
        self.k_max = k_max
        self.vaccination_rate = vaccination_rate
        self.vaccination_effect = vaccination_effect
        self.vaccination = vaccination

    def run(self):
        '''
        runs the system for the given parameter settings
        prints necessary time series plots
        '''
        self.initialize()
        for step in range(self.nr_timestep):
            self.update()
            self.observe()
        # plot the results for this parameter setting
        t = np.linspace(0,self.h*self.nr_timestep,self.nr_timestep+1)

        S = []
        I = []
        for step in range(self.nr_timestep+1):
            sum_S = 0
            sum_I = 0
            for k in range(1,self.k_max+1):
                sum_S += self.history['S_u'+str(k)][step] + self.history['S_v'+str(k)][step]
                sum_I += self.history['I_u'+str(k)][step] + self.history['I_v'+str(k)][step]
            S.append(sum_S)
            I.append(sum_I)
        
        pylab.plot(t, S, 'g-', label='S')
        pylab.plot(t, I, 'r-', label='I')
        pylab.ylim(0,1)
        pylab.xlabel('t')
        pylab.ylabel('fraction of population')
        pylab.title('k_max='+str(self.k_max)+ ' rho='+str(self.vaccination_effect)+ ' vacpolicy='+self.vaccination)
        pylab.legend(loc='upper right') 
        pylab.savefig('q2 k_max='+str(self.k_max)+ ' rho='+str(self.vaccination_effect)+ ' vacpolicy = '+self.vaccination+'.png')
        pylab.close()
        #pylab.show()

    def initialize(self):
        '''
        '''
        # pk dist 
        self.pk_dist = get_pk_dist(self.k_max)

        # a dict to keep track of current population
        self.current = {}
        for k in range(1,self.k_max+1):
            self.current['I_u'+str(k)] = self.population[k-1]
            self.current['S_u'+str(k)] = self.pk_dist[k-1] - self.population[k-1]

        # vaccination
        if self.vaccination == 'random':
            for k in range(1,self.k_max+1):
                self.current['I_v'+str(k)] = self.current['I_u'+str(k)]*self.vaccination_rate
                self.current['I_u'+str(k)] -= self.current['I_v'+str(k)]
                self.current['S_v'+str(k)] = self.current['S_u'+str(k)]*self.vaccination_rate
                self.current['S_u'+str(k)] -= self.current['S_v'+str(k)]

        elif self.vaccination == 'highest':
            vaccinated_sum = 0.0
            vacc_done = False
            for k in range(self.k_max,0,-1):#start vaccinating from the highest degree nodes
                if vacc_done == False:
                    if vaccinated_sum + self.current['I_u'+str(k)] + self.current['S_u'+str(k)] < 0.4: # vaccinate completely
                        self.current['I_v'+str(k)] = self.current['I_u'+str(k)]
                        self.current['I_u'+str(k)] = 0.0
                        self.current['S_v'+str(k)] = self.current['S_u'+str(k)]
                        self.current['S_u'+str(k)] = 0.0
                        vaccinated_sum += self.current['I_v'+str(k)] + self.current['S_v'+str(k)]
                    else: # vaccinate the remainder (slighlty prefer vaccinating I, no real reason)
                        vac_amount = 0.4 - vaccinated_sum
                        self.current['I_v'+str(k)] = self.current['I_u'+str(k)]
                        self.current['I_u'+str(k)] = 0.0
                        vac_amount -= self.current['I_v'+str(k)]
                        self.current['S_v'+str(k)] = vac_amount
                        self.current['S_u'+str(k)] -= self.current['S_v'+str(k)]
                        vacc_done = True
                else:
                    self.current['I_v'+str(k)] = 0.0
                    self.current['S_v'+str(k)] = 0.0

        elif self.vaccination == 'no_vaccine':
            for k in range(1,self.k_max+1):
                self.current['I_v'+str(k)] = 0.0
                self.current['S_v'+str(k)] = 0.0
            
        else:
            exit()

        # a dict to record keeping
        self.history = {}
        for k in range(1,self.k_max+1):
            self.history['I_u'+str(k)] = [ self.current['I_u'+str(k)] ]
            self.history['I_v'+str(k)] = [ self.current['I_v'+str(k)] ]
            self.history['S_u'+str(k)] = [ self.current['S_u'+str(k)] ]
            self.history['S_v'+str(k)] = [ self.current['S_v'+str(k)] ]

        # constants
        self.constant = {}
        for k in range(1,self.k_max+1):
            self.constant['p_u'+str(k)] = self.current['I_u'+str(k)] + self.current['S_u'+str(k)]
            self.constant['p_v'+str(k)] = self.current['I_v'+str(k)] + self.current['S_v'+str(k)]

    def observe(self):
        for k in range(1,self.k_max+1):
            self.history['I_u'+str(k)] += [ self.current['I_u'+str(k)] ]
            self.history['I_v'+str(k)] += [ self.current['I_v'+str(k)] ]
            self.history['S_u'+str(k)] += [ self.current['S_u'+str(k)] ]
            self.history['S_v'+str(k)] += [ self.current['S_v'+str(k)] ]

    def update(self):
        '''
        '''
        derivative = {}
        for k in range(1,self.k_max+1):
            derivative['I_u'+str(k)] = self.derivative_I_u(k)
            derivative['I_v'+str(k)] = self.derivative_I_v(k)
        for k in range(1,self.k_max+1):
            self.current['I_u'+str(k)] += self.h * derivative['I_u'+str(k)]
            self.current['I_v'+str(k)] += self.h * derivative['I_v'+str(k)]
            # sanity checks
            if self.current['I_u'+str(k)] < 0:
                self.current['I_u'+str(k)] = 0.0
            elif self.current['I_u'+str(k)] > 1:
                self.current['I_u'+str(k)] = 1.0
            if self.current['I_v'+str(k)] < 0:
                self.current['I_v'+str(k)] = 0.0
            elif self.current['I_v'+str(k)] > 1:
                self.current['I_v'+str(k)] = 1.0

            self.current['S_u'+str(k)] = self.constant['p_u'+str(k)] - self.current['I_u'+str(k)]
            self.current['S_v'+str(k)] = self.constant['p_v'+str(k)] - self.current['I_v'+str(k)]

    def derivative_I_v(self, k):
        sum_numerator = 0.0
        sum_denominator = 0.0
        for k_ in range(1,self.k_max+1):
            sum_numerator += k_ * (self.current['I_u'+str(k_)]+self.current['I_v'+str(k_)])
            sum_denominator += k_ * (self.constant['p_u'+str(k_)]+self.constant['p_v'+str(k_)])
        theta_vx = sum_numerator / sum_denominator
        d_I_v = (1-self.vaccination_effect) * self.beta * k * (self.constant['p_v'+str(k)] - self.current['I_v'+str(k)]) * theta_vx \
                - self.alpha * self.current['I_v'+str(k)]  
        return d_I_v
        
    def derivative_I_u(self, k):
        sum_numerator = 0.0
        sum_denominator = 0.0
        for k_ in range(1,self.k_max+1):
            sum_numerator += k_ * self.current['I_u'+str(k_)]
            sum_denominator += k_ * (self.constant['p_u'+str(k_)]+self.constant['p_v'+str(k_)])
        theta_uu = sum_numerator / sum_denominator
        sum_numerator = 0.0
        for k_ in range(1,self.k_max+1):
            sum_numerator += k_ * self.current['I_v'+str(k_)]
        theta_uv = sum_numerator / sum_denominator
        d_I_u = self.beta * k * (self.constant['p_u'+str(k)] - self.current['I_u'+str(k)]) * theta_uu \
                + (1-self.vaccination_effect) * self.beta * k * (self.constant['p_u'+str(k)] - self.current['I_u'+str(k)]) * theta_uv \
                - self.alpha * self.current['I_u'+str(k)]  
        return d_I_u

def main():
    K_MAX = 20
    BETA = 0.3
    ALPHA = 1
    VAC_RATE = 0.4
    pk_dist = get_pk_dist(K_MAX)
    population = pk_dist * 0.01
    for rho in range(6):
        print(rho*0.2)
        model = CONFIGURATION_MODEL(alpha=ALPHA, beta=BETA, population=population, 
                                    h=0.01, nr_timestep=1000, k_max=K_MAX, vaccination_rate=VAC_RATE, vaccination_effect=rho*0.2,
                                    vaccination='highest')
        model.run()
        model = CONFIGURATION_MODEL(alpha=ALPHA, beta=BETA, population=population, 
                                    h=0.01, nr_timestep=1000, k_max=K_MAX, vaccination_rate=VAC_RATE, vaccination_effect=rho*0.2,
                                    vaccination='random')
        model.run()
        model = CONFIGURATION_MODEL(alpha=ALPHA, beta=BETA, population=population, 
                                    h=0.01, nr_timestep=1000, k_max=K_MAX, vaccination_rate=VAC_RATE, vaccination_effect=0.0,
                                    vaccination='no_vaccine')
        model.run()

if __name__ == '__main__':
    main()
