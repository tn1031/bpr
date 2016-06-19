# coding: utf-8

import sys, time
import numpy as np
import scipy.sparse as sp

from MFbpr import MFbpr

def load_data(ratingFile, testRatio=0.1):
    user_count = item_count = 0
    ratings = []
    for line in open(ratingFile):
        arr = line.strip().split()
        user_id = int(arr[0])
        item_id = int(arr[1])
        score = float(arr[2])
        timestamp = long(arr[3])
        ratings.append((user_id, item_id, score, timestamp))
        user_count = max(user_count, user_id)
        item_count = max(item_count, item_id)    
    user_count += 1
    item_count += 1
    
    ratings = sorted(ratings, key=lambda x: x[3])   # sort by timestamp
    
    test_count = int(len(ratings) * testRatio)
    count = 0
    trainMatrix = sp.lil_matrix((user_count, item_count))
    testRatings = []
    for rating in ratings:
        if count < len(ratings) - test_count:
            trainMatrix[rating[0], rating[1]] = 1
        else:
            testRatings.append(rating)
        count += 1
    
    newUsers = set([])
    newRatings = 0
    
    for u in xrange(user_count):
        if trainMatrix.getrowview(u).sum() == 0:
            newUsers.add(u)
    for rating in ratings:
        if rating[0] in newUsers:
            newRatings += 1
    
    sys.stderr.write("Data\t{}\n".format(ratingFile))
    sys.stderr.write("#Users\t{}, #newUser: {}\n".format(user_count, len(newUsers)))
    sys.stderr.write("#Items\t{}\n".format(item_count))
    sys.stderr.write(
        "#Ratings\t {} (train), {}(test), {}(#newTestRatings)\n".format(
            trainMatrix.sum(),  len(testRatings), newRatings))
    
    return trainMatrix, testRatings

def evaluate_model_online(model, name, interval):
    start = time.time()
    model.evaluateOnline(testRatings, interval)
    sys.stderr.write("{}\t <hr, ndcg, prec>:\t {}\t {}\t {} [{}]\n".format( 
                     name, np.mean(model.hits), np.mean(model.ndcgs), np.mean(model.precs),
                     time.time() - start))



if __name__ == "__main__":
    # data
    trainMatrix, testRatings = load_data('yelp.rating')

    # settings
    topK = 100
    factors = 64
    maxIter = 10
    maxIterOnline = 1
    lr = 0.01
    adaptive = False
    init_mean = 0.0
    init_stdev = 0.1
    reg = 0.01
    showProgress = False
    showLoss = True

    bpr = MFbpr(trainMatrix, testRatings, topK, factors, maxIter, lr, adaptive, reg, init_mean, init_stdev, showProgress, showLoss)
    bpr.buildModel()

    bpr.maxIterOnline = maxIterOnline;
    evaluate_model_online(bpr, "BPR", 1000);




