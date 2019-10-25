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
    mean = 5
    sigma = 1
    mean1 = 0
    sigma1 = 0.2
    n = 1000
    xi = np.random.normal(mean, sigma, 500)
    yi = np.random.normal(mean1, sigma1, 500)
    data = np.concatenate((xi,yi), axis = 0)
    h=0.1
    prob = []
    j = tuple(numeric_range(min(data),max(data),0.001))
    lenN = len(data)
    kdeobj = mykde()
    kdeobj.kde(prob,data,lenN,h,j)
    plt.plot(prob)
    plt.show()
    plt.hist(data)
    
if __name__ == "__main__":
	main()
