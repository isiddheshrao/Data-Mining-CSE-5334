# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:38:51 2019

@author: Siddhesh Rao
"""

import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self, alpha=0.001, epochs=100000):
        self.alpha = alpha
        self.weights = np.ones(3)
        self.epochs = epochs
        self.loss_threshold = 0.001
        self.current_loss = float('inf')
        self.previous_loss = float('inf')
        self.training_converged = False
        self.iteration_count = 0
        self.total_cost = []
        self.tot_iter = []
        self.norm_current_loss = []
        self.norm_grad=[]

    def activation_sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cross_entropy_loss(self, predicted, actual):
        self.cost = (-actual * np.log(predicted) - (1 - actual) * np.log(1 - predicted)).mean()
        self.total_cost.append(self.cost)
        return self.cost

    def learn_per_instance(self, X, y):
        net_val = np.dot(X, self.weights)
        prediction = self.activation_sigmoid(net_val)
        self.gradient = np.dot(X.T, (prediction - y))
        self.weights -= self.alpha * self.gradient
        self.current_loss = self.cross_entropy_loss(prediction, y)
        self.norm_grad.append(abs(self.gradient[0])+abs(self.gradient[1])+abs(self.gradient[2]))
        

        if self.iteration_count % 100 == 0:
            print("Cross - Entropy loss at iteration %s: %s" % (self.iteration_count + 1, self.current_loss))

        if (self.gradient == np.zeros(X.shape[0])).all():
            print("Gradient is zero!")
            print("total no. of iterations run: ", self.iteration_count + 1)
            self.training_converged = True

        if self.current_loss < self.loss_threshold:
            print("Loss optimized is less than threshold!")
            print("total no. of iterations run: ", self.iteration_count + 1)
            self.training_converged = True
        
        self.norm_current_loss.append(abs(self.current_loss-self.previous_loss))

        self.previous_loss = self.current_loss
        self.iteration_count += 1
        self.store_iter = self.iteration_count
        self.tot_iter.append(self.store_iter)
        #print ("iter",self.tot_iter)
        return self.norm_current_loss, self.tot_iter, self.norm_grad

    def learn(self, X, target):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        for epoch in range(self.epochs):
            print("\n\nEpoch - ", epoch + 1)
            for i in range(X.shape[0]):
                self.learn_per_instance(X[i, :], target[i])
                if self.training_converged:
                    break

            if self.training_converged:
                break

    def predict(self, X):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return np.round(self.activation_sigmoid(np.dot(X, self.weights)))
    


def generate_data(mean, variance, count):
    return np.random.multivariate_normal(mean, variance, count)


def calculateAccuracy(predicted_y, test_y):
    predicted_y = predicted_y.tolist()
    test_y = test_y.tolist()

    count = 0
    for i in range(len(predicted_y)):
        if predicted_y[i] == test_y[i]:
            count += 1

    return (count / len(predicted_y)) * 100



max_epochs = 100000
x1 = generate_data([1, 0], [[1, 0.75], [0.75, 1]], 500)
x2 = generate_data([0, 1.5], [[1, 0.75], [0.75, 1]], 500)
X = np.vstack((x1, x2)).astype(np.float32)
y = np.hstack((np.zeros(500), np.ones(500)))

test_x1 = generate_data([1, 0], [[1, 0.75], [0.75, 1]], 500)
test_x2 = generate_data([0, 1.5], [[1, 0.75], [0.75, 1]], 500)
test_X = np.vstack((test_x1, test_x2)).astype(np.float32)
test_y = np.hstack((np.zeros(500), np.ones(500)))

print("\n\nLearning rate (Alpha): 1\nTotal Epochs: 100000")
LR = LogisticRegression(alpha=1, epochs=max_epochs)
LR.learn(X, y)
print("Final Weights: ", LR.weights)
predicted_y = LR.predict(test_X)
accuracy = calculateAccuracy(predicted_y, test_y)
print("Accuracy: ", accuracy)
f, ax = plt.subplots(1, figsize=(5, 5))
ax.set_title("Norm Gradient")
ax.plot(LR.tot_iter, LR.norm_grad, 'r', label=r'$\alpha = 1$') 
ax.set_xlabel('Iterations')
ax.set_ylabel('Gradient')
ax.legend();
f, ax = plt.subplots(1, figsize=(5, 5))
plt.figure(figsize=(8, 8))
ax.set_title("Validation loss")
ax.plot(range(0, len(LR.total_cost)), LR.total_cost, 'r', label=r'$\alpha = 1$') 
ax.set_xlabel('Iterations')
ax.set_ylabel('Training Error')
ax.legend();
plt.figure(figsize=(5, 5))
x_d = np.linspace(-3, 4, 50)
y_d = (LR.weights[0]*x_d)/LR.weights[1]
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=.4)
plt.plot(x_d, y_d)
plt.show()
print("==================================================")

print("\n\nLearning rate (Alpha): 0.1\nTotal Epochs: 100000")
LR = LogisticRegression(alpha=0.1, epochs=max_epochs)
LR.learn(X, y)
print("Final Weights: ", LR.weights)
predicted_y = LR.predict(test_X)
accuracy = calculateAccuracy(predicted_y, test_y)
print("Accuracy: ", accuracy)
f, ax = plt.subplots(1, figsize=(5, 5))
ax.set_title("Norm Gradient")
ax.plot(LR.tot_iter, LR.norm_grad, 'r', label=r'$\alpha = 0.1$') 
ax.set_xlabel('Iterations')
ax.set_ylabel('Gradient')
ax.legend();
f, ax = plt.subplots(1, figsize=(5, 5))
plt.figure(figsize=(8, 8))
ax.set_title("Validation loss")
ax.plot(range(0, len(LR.total_cost)), LR.total_cost, 'r', label=r'$\alpha = 0.1$') 
ax.set_xlabel('Iterations')
ax.set_ylabel('Training Error')
ax.legend();
plt.figure(figsize=(5, 5))
x_d = np.linspace(-3, 4, 50)
y_d = (LR.weights[0]*x_d)/LR.weights[1]
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=.4)
plt.plot(x_d, y_d)
plt.show()
print("==================================================")

print("\n\nLearning rate (Alpha): 0.01\nTotal Epochs: 100000")
LR = LogisticRegression(alpha=0.01, epochs=max_epochs)
LR.learn(X, y)
print("Final Weights: ", LR.weights)
predicted_y = LR.predict(test_X)
accuracy = calculateAccuracy(predicted_y, test_y)
print("Accuracy: ", accuracy)
f, ax = plt.subplots(1, figsize=(5, 5))
ax.set_title("Norm Gradient")
ax.plot(LR.tot_iter, LR.norm_grad, 'r', label=r'$\alpha = 0.01$') 
ax.set_xlabel('Iterations')
ax.set_ylabel('Gradient')
ax.legend();
f, ax = plt.subplots(1, figsize=(5, 5))
plt.figure(figsize=(8, 8))
ax.set_title("Validation loss")
ax.plot(range(0, len(LR.total_cost)), LR.total_cost, 'r', label=r'$\alpha = 0.01$') 
ax.set_xlabel('Iterations')
ax.set_ylabel('Training Error')
ax.legend();
plt.figure(figsize=(5, 5))
x_d = np.linspace(-3, 4, 50)
y_d = (LR.weights[0]*x_d)/LR.weights[1]
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=.4)
plt.plot(x_d, y_d)
plt.show()
print("==================================================")

print("\n\nLearning rate (Alpha): 0.001\nTotal Epochs: 100000")
LR = LogisticRegression(alpha=0.001, epochs=max_epochs)
LR.learn(X, y)
print("Final Weights: ", LR.weights)
predicted_y = LR.predict(test_X)
accuracy = calculateAccuracy(predicted_y, test_y)
print("Accuracy: ", accuracy)
f, ax = plt.subplots(1, figsize=(5, 5))
ax.set_title("Norm Gradient")
ax.plot(LR.tot_iter, LR.norm_grad, 'r', label=r'$\alpha = 0.001$') 
ax.set_xlabel('Iterations')
ax.set_ylabel('Gradient')
ax.legend();
f, ax = plt.subplots(1, figsize=(5, 5))
plt.figure(figsize=(8, 8))
ax.set_title("Validation loss")
ax.plot(range(0, len(LR.total_cost)), LR.total_cost, 'r', label=r'$\alpha = 0.001$') 
ax.set_xlabel('Iterations')
ax.set_ylabel('Training Error')
ax.legend();
plt.figure(figsize=(5, 5))
x_d = np.linspace(-3, 4, 50)
y_d = (LR.weights[0]*x_d)/LR.weights[1]
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=.4)
plt.plot(x_d, y_d)
plt.show()
print("==================================================")

plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=.4)
plt.show()