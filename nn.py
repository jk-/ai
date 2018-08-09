
################
## DATA IS NOT SCALED PROPERLY TO MATCH SIGMOID!
################


# this 3 layer NN takes an input of 
# inverse fed balance sheet y/y, consumer ESI and europe BS y/y
#
# input = [fbs,con,ebs]
# output = eurusd target
#
# 3 input nodes, 4 hidden nodes, 1 output node
#

import sys
import numpy as np
from numpy import genfromtxt

np.random.seed(1)
learning_rate = 0.5

def inputpredict():
    r,g,b = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    features = np.array([r,g,b])
    predict(features)

def masspredict():
    trainingData = genfromtxt('predict.csv', delimiter=',')[1::]
    features = trainingData
    predict(features)


def predict(features):
    # taken from training
    syn0 = [[-0.1970442,0.37423284, -1.02183979, -0.45759824],
        [-1.34580949, -1.46633727, -1.3173524,5.30540803],
        [-0.22598985, -0.00935176, -0.16247903,  0.31048998]]
    syn1 = [[ 3.54829253],
        [ 4.79201775],
        [ 3.2661089 ],
        [13.1659753 ]]

    l0 = features
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    print(l2)

# relu
def relu(x, d=False):
    if (d):
        x[x<=0] = 0
        x[x>0] = 1
        return x
    return np.maximum(x, 0)

# activation function
def sigmoid(x, d=False):
    if (d):
        return (x*(1-x))
    return 1/(1+np.exp(-x))

def scale_range (input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input

# trainign is basically creating an accurate representation of
# synapses / weights so the prediction can be close as possible
def train():
    trainingData = genfromtxt('data.csv', delimiter=',')[1::]
    features = trainingData[:len(trainingData),:3]
    labels = trainingData[:len(trainingData),3::]

    # weights / synapses / connections
    # syn0 is the weights for features -> hidden layers
    # syn1 is the weights for hidden layers -> outputs
    # 3,4 is the 3 inputs to 4 hidden nodes
    # 4,1 is the 4 hidden nodes into the 1 output node
    syn0 = 2 * np.random.random((3,4)) - 1
    syn1 = 2 * np.random.random((4,1)) - 1

    l2_sqr_err = 100;
    itt = 0;

    # loop 60000 iterations so the weights have enough
    # times to adjust to lowering the error rate towards zero
    while (l2_sqr_err > .1 and itt <= 100000):
        itt += 1

        # our first layer are the inputs r, g, b
        l0 = features

        # below we "feed forward" through the network

        # our second layer are the hidden layers
        # we take the dot product of inputs and synapses
        # and run them through the activation function
        l1 = sigmoid(np.dot(l0, syn0))

        # we take the hidden nodes and generate our third layer
        # the output layer, and run them through the activation
        # function to normalize probabilities
        l2 = sigmoid(np.dot(l1, syn1))

        # backward propagation where we update our weights

        # this is the total error of our expected output (targets)
        # and the current output
        l2_error = labels - l2
        
        # we need to take the mean squared error to find the actuall 
        # error rate
        l2_sqr_err = np.mean(np.abs(l2_error))

        if (itt % 10000) == 0:
            print ('epoch', itt ,'Error Rate:', str(l2_sqr_err))

        # here we find the delta between our output layers
        # and the deriviate of the weights
        l2_delta = l2_error * learning_rate * sigmoid(l2, d=True)

        # we find the hidden layer deltas by taking the dot product of
        # the output deltas and output synapse weights
        l1_error = l2_delta.dot(syn1.T)

        # now we go back even further and use the hidden layer
        # erorr rates and find the hidden layer deltas
        l1_delta = l1_error * learning_rate * sigmoid(l1, d=True)

        # now that we have the deltas between the expected and output
        # we can update our synapses/weights
        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.T.dot(l1_delta)


    #print('syn0 weights')
    print(syn0)
    #print('syn1 weights')
    print(syn1)
    #print('expected results after training')
    #print(labels)
    #print(np.around(l2,decimals=1))

masspredict()