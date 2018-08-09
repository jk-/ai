### Notes: 
#   One issue.. the feed forward net we created only works for labels with 0/1
#
#   Solution to this problem: since our input of 255, 255, 255
#   always produced a max of 1 in our sigmoid, it was unable to
#   updates the weights properly. I noramlized the inputs to X/255
#   so it was a ratio between 0 and 1 which allowed our sigmoid to progress
#


# this 3 layer NN takes an input of R,G,B values
# and tells you if the text should be black or white
#
# input = [r,g,b]
# output = [0|1]
#
# 3 input nodes, 4 hidden nodes, 2 output nodes
#
#   inputpredict to take a single value
#        rgb.py 255 255 255
#
#   masspredict to take a large array of inputs
#

import sys
import numpy as np

np.random.seed(1)
learning_rate = 0.5

def inputpredict():
    r,g,b = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    features = np.array([r,g,b])
    predict(features)

def masspredict():
    features = np.array([[224,168,93],
        [130,156,65],
        [121,227,29],
        [71,179,198],
        [224,3,31],
        [187,45,110],
        [29,247,56],
        [194,44,203],
        [62,242,97],
        [212,18,17],
        [100,74,253],
        [223,204,232],
        [221,124,17],
        [180,191,95],
        [93,218,81],
        [212,130,90],
        [112,2,234],
        [24,1,182],
        [66,66,236],
        [221,169,39],
        [219,130,28],
        [236,106,180],
        [215,179,162],
        [245,80,16],
        [246,230,215]])
    predict(features)


def predict(features):
    features = normalize(features)

    # taken from training
    syn0 = [[-2.08961435,-0.47214912,-2.13670542,-2.13880068],
        [-4.24999767,-0.78078341,-4.3421069,-4.34619838],
        [-0.45586479,1.69207576,-0.4438203,-0.44328622]]
    syn1 = [[58.65008776],
        [-8.93854651],
        [64.58167214],
        [64.85702617]]

    l0 = features
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    l2 = np.round(l2)
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

def normalize(x):
    return x/225;

# trainign is basically creating an accurate representation of
# synapses / weights so the prediction can be close as possible
def train():
    # features = x = inputs
    features = np.array([
        [247,30,227],
        [154,46,171],
        [1,96,85],
        [251,97,168],
        [231,120,240],
        [84,0,141],
        [32,78,97],
        [48,58,206],
        [151,225,177],
        [7,76,218],
        [207,133,202],
        [220,7,59],
        [245,188,131],
        [32,222,54],
        [18,105,6],
        [145,39,160],
        [139,114,40],
        [93,72,69],
        [165,197,119],
        [24,188,6],
        [120,177,212],
        [145,205,24],
        [69,236,85],
        [64,28,121],
        [55,64,69],
        [66,56,219],
        [75,99,222],
        [79,81,135],
        [179,5,208],
        [105,27,187],
        [74,182,17],
        [166,36,169],
        [4,254,174],
        [125,203,38],
        [104,108,87],
        [215,210,181],
        [157,25,200],
        [198,79,85],
        [121,226,80],
        [109,118,0],
        [91,110,169],
        [152,232,207],
        [128,46,62],
        [185,107,8],
        [252,165,178],
        [16,40,101],
        [134,47,7],
        [95,215,107],
        [130,210,20],
        [180,132,173],
        [14,146,206],
        [216,110,116],
        [31,17,61],
        [150,40,94],
        [12,101,213],
        [60,149,177],
        [250,182,71],
        [18,42,10],
        [39,77,142],
        [12,94,91],
        [214,87,40],
        [185,202,224],
        [49,34,173],
        [152,192,171],
        [148,191,186],
        [147,157,169],
        [55,100,45],
        [13,164,36],
        [218,79,118],
        [204,246,41],
        [201,108,147],
        [186,56,103],
        [150,196,205],
        [26,129,66],
        [75,218,206],
        [73,227,253],
        [116,50,147],
        [112,245,186]
    ])

    # labels = y = outputs = "targets"
    labels = np.array([
        [0],
        [1],
        [1],
        [0],
        [0],
        [1],
        [1],
        [1],
        [0],
        [1],
        [0],
        [1],
        [0],
        [0],
        [1],
        [1],
        [1],
        [1],
        [0],
        [1],
        [0],
        [0],
        [0],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [0],
        [1],
        [0],
        [1],
        [1],
        [0],
        [1],
        [1],
        [0],
        [1],
        [1],
        [0],
        [1],
        [1],
        [0],
        [1],
        [1],
        [0],
        [0],
        [0],
        [0],
        [0],
        [1],
        [1],
        [1],
        [0],
        [0],
        [1],
        [1],
        [1],
        [0],
        [0],
        [1],
        [0],
        [0],
        [0],
        [1],
        [0],
        [0],
        [0],
        [1],
        [1],
        [0],
        [1],
        [0],
        [0],
        [1],
        [0]
    ])

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
    while (l2_sqr_err > .1):
        itt += 1

        # our first layer are the inputs r, g, b
        l0 = features
        l0 = normalize(l0)

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
        l2_sqr_err = np.mean(np.abs(l2_error))

        # we need to take the mean squared error to find the actuall 
        # error rate
        if (itt % 10000) == 0:
            print ('Error Rate: ', str(l2_sqr_err))

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
    #print(syn0)
    #print('syn1 weights')
    #print(syn1)
    #print('expected results after training')
    #print(labels)
    #print(np.around(l2,decimals=1))

inputpredict()