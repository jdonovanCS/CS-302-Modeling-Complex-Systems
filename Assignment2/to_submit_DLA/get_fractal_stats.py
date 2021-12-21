import numpy as np
import matplotlib.pyplot as plt
import sys

def main(path_to_txt):
    f = open(path_to_txt, "r")
    r = []
    N = []
    for line in f:
        r_, N_ = line.split(' ')
        r.append(1. / float(r_))
        N.append(float(N_))
    r = np.log( np.asarray(r) )
    N = np.log( np.asarray(N) )

    # fit line
    slope, b = np.polyfit(r,N,1)
    # plot it 
    plt.plot(r, N, 'o')
    plt.plot(r, r*slope+b)
    plt.xlabel("log(r)")
    plt.ylabel("log(N)")
    plt.title("d="+str(slope))
    plt.savefig(path_to_txt.split('.')[0]+'_fractalPlot.png')
   

if __name__=='__main__':
    main(sys.argv[1])
   








