# coding: utf-8

import sys, time
import numpy as np
import scipy.sparse as sp

from utils import Utils

class Recommender(object):
    def __init__(self, trainMatrix, testRatings, topK=100):
        self.trainMatrix = trainMatrix.copy()
        self.testRatings = testRatings
        self.topK = topK
        
        self.userCount = self.trainMatrix.shape[0]
        self.itemCount = self.trainMatrix.shape[1]
        
        self.maxIterOnline = 1
        self.hits = None
        self.ndcgs = None
        self.precs = None
        
        self.ignoreTrain = False
        
    def _showProgress(self, itr, start, testRatings):
        end_itr = time.time()
        if self.userCount == len(testRatings):
            # leave-1-out eval
            self.evaluate(testRatings)
        else:
            # global split
            self.evaluateOnline(testRatings, 1000)
        end_eval = time.time()
        
        sys.stderr.write(
            "Iter={}[{}] <loss, hr, ndcg, prec>:\t {}\t {}\t {}\t {}\t [{}]\n".format(itr, 
                                                                                     end_itr - start, self.loss(), 
                                                                                     np.mean(self.hits), np.mean(self.ndcgs), np.mean(self.precs), 
                                                                                     end_eval - end_itr))

        
    def evaluateOnline(self, testRatings, interval):
        testCount = len(testRatings)
        self.hits = np.array([0.0]*testCount)
        self.ndcgs = np.array([0.0]*testCount)
        self.precs = np.array([0.0]*testCount)
        
        intervals = 10
        counts = [0] * (intervals + 1)
        hits_r = [0.0] * (intervals + 1)
        ndcgs_r = [0.0] * (intervals + 1)
        precs_r = [0.0] * (intervals + 1)
        
        updateTime = 0
        for i in xrange(testCount):
            if i > 0 and interval > 0 and i % interval == 0:
                # Check performance per interval:
                sys.stderr.write("{}: <hr, ndcg, prec> =\t {}\t {}\t {}\n".format(
                        i, np.sum(self.hits) / i, np.sum(self.ndcgs) / i, np.sum(self.precs) / i))
            # Evaluate model of the current test rating:
            rating = testRatings[i]
            res = self.evaluate_for_user(rating[0], rating[1])
            self.hits[i] = res[0]
            self.ndcgs[i] = res[1]
            self.precs[i] = res[2]
                
            # statisitcs for break down
            r = len(self.trainMatrix.getrowview(rating[0]).rows[0])
            r = intervals if r > intervals else r
            counts[r] += 1
            hits_r[r] += res[0]
            ndcgs_r[r] += res[1]
            precs_r[r] += res[2]
                
            # Update the model
            start = time.time()
            self.updateModel(rating[0], rating[1])
            updateTime += time.time() - start
                
        sys.stderr.write("Break down the results by number of user ratings for the test pair.\n")
        sys.stderr.write("#Rating\t Percentage\t HR\t NDCG\t MAP\n")
        for i in xrange(intervals+1):
            if counts[i] == 0:
                continue
            sys.stderr.write("{}\t {}%%\t {}\t {}\t {} \n".format(
                    i, float(counts[i])/testCount*100, 
                    hits_r[i] / counts[i], ndcgs_r[i] / counts[i], precs_r[i] / counts[i]))
        
        sys.stderr.write("Avg model update time per instance: {}\n".format(float(updateTime)/testCount))
        
    def evaluate(self, testRatings):
        self.hits = np.array([0.0] * self.userCount)
        self.ndcgs = np.array([0.0] * self.userCount)
        self.precs = np.array([0.0] * self.userCount)
        
        for rating in testRatings:
            u = rating[0]
            i = rating[1]
            res = self.evaluate_for_user(u, i)
            self.hits[u] = res[0]
            self.ndcgs[u] = res[1]
            self.precs[u] = res[2]
            
    
    def evaluate_for_user(self, u, gtItem):
        result = [0.0] * 3
        map_item_score = {}
        # Get the score of the test item first.
        maxScore = self.predict(u, gtItem)
        
        # Early stopping if there are topK items larger than maxScore.
        countLarger = 0
        for i in xrange(self.itemCount):
            score = self.predict(u, i)
            map_item_score[i] = score
            
            if score > maxScore:
                countLarger += 1
            if countLarger > self.topK:
                # early stopping
                return result
        
        # Selecting topK items (does not exclude train items).
        if self.ignoreTrain:
            rankList = Utils.TopKeysByValue(map_item_score, self.topK, self.trainMatrix.getrowview(u).rows[0])
        else:
            rankList = Utils.TopKeysByValue(map_item_score, self.topK, None)

        result[0] = self.getHitRatio(rankList, gtItem)
        result[1] = self.getNDCG(rankList, gtItem)
        result[2] = self.getPrecision(rankList, gtItem)
        
        return result
    
    def getHitRatio(self, rankList, gtItem):
        for item in rankList:
            if item == gtItem:
                return 1
        return 0
    
    def getNDCG(self, rankList, gtItem):
        for i, item in enumerate(rankList):
            if item == gtItem:
                return np.log(2) / np.log(i+2)
        return 0
    
    def getPrecision(self, rankList, gtItem):
        for i, item in enumerate(rankList):
            if item == gtItem:
                return 1.0 / (i+1)
        return 0

