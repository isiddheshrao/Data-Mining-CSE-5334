# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 20:02:33 2019

@author: Siddhesh Rao
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 19:49:44 2019

@author: Siddhesh Rao
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 14:46:46 2019

@author: Siddhesh Rao
"""

import numpy as np
import matplotlib.pyplot as plt
from more_itertools import numeric_range

class mykde:
    def kde(self,prob,data,lenN,h,j):
        for x in j:
            sum_k = 0
            for i in data:
                u = (x - i)/h
                if abs(u) <=0.5:
                    k=1
                else:
                    k=0
                sum_k= sum_k+k
            p=sum_k*(1/lenN*h)
            prob.append(p)
        print (prob, sum_k)
        return


def main():
    #Initialize Data
    mean = [1,0]
    sigma = [[0.9,0.4],[0.4,0.9]]
    mean1 = [0,1.5]
    sigma1 = [[0.9,0.4],[0.4,0.9]]
    n = 1000
    x,y = np.random.multivariate_normal(mean, sigma, 500).T
    X = np.array(list(zip(x,y)))
    a,b = np.random.multivariate_normal(mean1, sigma1, 500).T
    Y = np.array(list(zip(a,b)))
    data = np.concatenate((X,Y), axis = 0)
    h=0.1
    prob = []
    j = np.linspace(min(data),max(data),0.001)
    lenN = len(data)
    kdeobj = mykde()
    kdeobj.kde(prob,data,lenN,h,j)
    plt.plot(prob)
    plt.show()
    plt.hist(data)
    
if __name__ == "__main__":
	main()