# coding: utf-8

import sys, time
import random
import numpy as np
import scipy.sparse as sp

from recommender import Recommender

class MFbpr(Recommender):
    def __init__(self, trainMatrix, testRatings, topK=100, 
                 factors=10, maxIter=500, lr=0.01, adaptive=False, reg=0.01, 
                 init_mean=0.0, init_stdev=0.1, showProgress=False, showLoss=True):
        super(MFbpr, self).__init__(trainMatrix, testRatings, topK)
        
        # Model priors to set.
        self.factors = factors  # number of latent factors.
        self.maxIter = maxIter  # maximum iterations.
        self.reg = reg  # regularization parameters
        self.lr = lr
        self.adaptive = adaptive
        self.init_mean = init_mean  # Gaussian mean for init V
        self.init_stdev = init_stdev  # Gaussian std-dev for init V
        self.showProgress = showProgress
        self.showLoss = showLoss
        
        self.onlineMode = 'u'

        # Init model parameters
        self.U = np.random.normal(
            self.init_mean, self.init_stdev, self.userCount*self.factors).reshape((self.userCount, self.factors))
        self.V = np.random.normal(
            self.init_mean, self.init_stdev, self.itemCount*self.factors).reshape((self.itemCount, self.factors))
        

    def setUV(self, U, V):
        self.U = np.copy(U)
        self.V = np.copy(V)
        
    def buildModel(self):
        loss_pre = sys.float_info.max
        nonzeros = self.trainMatrix.nnz
        hr_prev = 0.0
        sys.stderr.write("Run for BPR. \n")
        for itr in xrange(self.maxIter):
            start = time.time()
            
            # Each training epoch
            for s in xrange(nonzeros):
                # sample a user
                u = np.random.randint(self.userCount)
                itemList = self.trainMatrix.getrowview(u).rows[0]
                if len(itemList) == 0:
                    continue
                # sample a positive item
                i = random.choice(itemList)
                
                # One SGD step update
                self.update_ui(u, i)
            
            # Show progress
            if self.showProgress:
                self._showProgress(itr, start, self.testRatings)
            
            # Show loss
            if self.showLoss:
                loss_pre = self._showLoss(itr, start, loss_pre)
                
            if self.adaptive:
                if not self.showProgress:
                    self.evaluate(self.testRatings)
                hr = np.mean(self.ndcgs)
                self.lr = self.lr * 1.05 if hr > hr_prev else self.lr * 0.5
                hr_prev = hr

    # Run model for one iteration
    def runOneIteration(self):
        nonzeros = self.trainMatrix.nnz
        # Each training epoch
        for s in xrange(nonzeros):
            u = np.random.randint(self.userCount)
            itemList = self.trainMatrix.getrowview(u).rows[0]

            if len(itemList) == 0:
                continue
            # sample a positibe item
            i = random.choice(itemList)
            
            # One SGD update
            self.update_ui(u, i)

    def update_ui(self, u, i):
        # sample a negative item(uniformly random)
        j = np.random.randint(self.itemCount)
        while self.trainMatrix[u, j] != 0:
            j = np.random.randint(self.itemCount)
            
        # BPR update rules
        y_pos = self.predict(u, i)  # target value of positive instance
        y_neg = self.predict(u, j)  # target value of negative instance
        mult = -self.partial_loss(y_pos - y_neg)
        
        for f in xrange(self.factors):
            grad_u = self.V[i, f] - self.V[j, f]
            self.U[u, f] -= self.lr * (mult * grad_u + self.reg * self.U[u, f])
                
            grad = self.U[u, f]
            self.V[i, f] -= self.lr * (mult * grad + self.reg * self.V[i, f])
            self.V[j, f] -= self.lr * (-mult * grad + self.reg * self.V[j, f])
        

    # Partial of the ln sigmoid function used by BPR
    def partial_loss(self, x):
        exp_x = np.exp(-x)
        return exp_x / (1.0 + exp_x)
    
    def _showLoss(self, itr, start, loss_pre):
        start1 = time.time()
        loss_cur = self.loss()
        symbol = "-" if loss_pre >= loss_cur else "+"
        sys.stderr.write(
            "Iter={} [{}]\t [{}]loss: {} [{}]\n".format(itr, 
                                                        start1 - start, symbol, loss_cur, 
                                                        time.time() - start1))
        return loss_cur

    #  Fast way to calculate the loss function
    def loss(self):
        L = self.reg * (np.sum(np.square(self.U)) + np.sum(np.square(self.V)))
        for u in xrange(self.userCount):
            l = 0
            for i in self.trainMatrix.getrowview(u).rows[0]:
                pred = self.predict(u, i)
                l += np.power(self.trainMatrix[u, i] - pred, 2)
            L += l
            
        return L

    def predict(self, u, i):
        return np.dot(self.U[u], self.V[i])

    def updateModel(self, u, item):
        self.trainMatrix[u, item] = 1
        
        # user retain
        itemList = self.trainMatrix.getrowview(u).rows[0]
        for itr in xrange(self.maxIterOnline):
            random.shuffle(itemList)
            
            for s in xrange(len(itemList)):
                # retrain for the user or for the (user, item) pair
                i = itemList[s] if self.onlineMode == 'u' else item
                self.update_ui(u, i)

