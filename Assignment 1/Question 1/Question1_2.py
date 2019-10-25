# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 23:26:23 2019

@author: Siddhesh Rao
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:26:21 2019

@author: Siddhesh Rao
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

class myKmeans:
    def __init__(self, tolerance = 0.001, max_iterations = 10000):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
    def calckmeans(self,data,k, cent):
        self.centroids = {}
        for i in range(k):
            self.centroids[i] = cent[i]
        for i in range(self.max_iterations):
            self.classes = {}
            for i in range(k):
                self.classes[i] = []
            for attr in data:
                distances = [np.linalg.norm(attr - self.centroids[centroid]) for centroid in self.centroids]
                classify = distances.index(min(distances))
                self.classes[classify].append(attr)
            previous = dict(self.centroids)
            for classify in self.classes:
                self.centroids[classify] = np.average(self.classes[classify], axis = 0)
            valid = True
            for centroid in self.centroids:
                original_centroid = previous[centroid]
                curr = self.centroids[centroid]
                if np.sum((curr - original_centroid)/original_centroid * 100.0) > self.tolerance:
                    valid = False
            if valid:
                break
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classify = distances.index(min(distances))

def main():
    mean = [1,0]
    mean1 = [0,1.5]
    cov = [[0.9, 0.4], [0.4, 0.9]]
    cov1 = [[0.9, 0.4], [0.4, 0.9]]
    cent = [[10, 10],[-10, -10]]
    x, y = np.random.multivariate_normal(mean, cov, 500).T
    X = np.array(list(zip(x,y)))
    a, b = np.random.multivariate_normal(mean1, cov1, 500).T
    Y = np.array(list(zip(a,b)))
    data = np.concatenate((X,Y), axis = 0)
    kmeans = myKmeans()
    kmeans.calckmeans(data, 2, cent)
    colors = 10*["r", "g", "c", "b", "k"]
    for classify in kmeans.classes:
        color = colors[classify]
        for attr in kmeans.classes[classify]:
            plt.scatter(attr[0], attr[1], color = color,s = 30)
    for centroid in kmeans.centroids:
        plt.scatter(kmeans.centroids[centroid][0], kmeans.centroids[centroid][1], s = 130, marker = "D", color = 'K')
    print ("Centroid Coordinates:")
    print (kmeans.centroids)
    plt.show()
    
if __name__ == "__main__":
	main()