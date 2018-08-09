# this 3 layer NN takes an input of R,G,B values
# and tells you if the text should be black or white
# input = [r,g,b]
# output = [0|1]

import sys
import numpy as np

np.random.seed(2)
learning_rate = 0.5

# activation function
def sigmoid(x, d=False):
    if (d):
        return (x*(1-x))
    return 1/(1+np.exp(-x))

def train():
    # features = x = inputs
    features = []

    # labels = y = outputs = "targets"
    labels = []

    # weights / synapses / connections
    # syn0 is the weights for features -> hidden layers
    # syn1 is the weights for hidden layers -> outputs
    # 3,4 is the 3 inputs to 4 hidden nodes
    # 4,1 is the 4 hidden nodes into the 1 output node
    syn0 = 2 * np.random.random((3,4)) - 1
    syn1 = 2 * np.random.random((4,1)) - 1

    # loop 60000 iterations so the weights have enough
    # times to adjust to lowering the error rate towards zero
    for j in xrange(60000):
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
        if (j % 5000):
            print 'Error Rate: ' + str(np.mean(np.abs(l2_error)))

        # here we find the delta between our output layers
        # and the deriviate of the weights
        l2_delta = l2_error * sigmoid(l2, d=True)

        # we find the hidden layer deltas by taking the dot product of
        # the output deltas and output synapse weights
        l1_error = l2_delta.dot(syn1.T)

        # now we go back even further and use the hidden layer
        # erorr rates and find the hidden layer deltas
        l1_delta = l1_error * sigmoid(l1, d=true)

        # now that we have the delates between the expected and output
        # we can update our synapses/weights
        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.T.dot(l1_delta)

'''
# inputs are R, G, B
r,g,b = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
inputs = [r,g,b]




inputs r,g,b

two hidden layers
h1
h2

two bias one for hidden, one for outputs
b1
b2

h1 is equal to the dot product of inputs and wh1 + b1 * 1
h2 is equal to the dot product of inputs and wh2 + b1 * 1

h1 and h2 run through sigmoid

two output layers

o1 is equal to h1 * wo1 + b2 * 1 -> sigmoid
o2 is equal to h2 * wo2 + b2 * 1-> sigmoid

eo1 = sqrErr ( target, output )
eo2 = sqrErr ( target, output )
err = eo1 + eo2

now we need to back propagate to adjust the weights and reduce the error


'''