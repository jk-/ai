# this 3 layer NN takes an input of R,G,B values
# and tells you if the text should be black or white
#
# input = [r,g,b]
# output = [0|1]
#
# 3 input nodes, 4 hidden nodes, 2 output nodes

import sys
import numpy as np

np.random.seed(2)
learning_rate = 0.5

# weights / synapses / connections
# syn0 is the weights for features -> hidden layers
# syn1 is the weights for hidden layers -> outputs
# 3,4 is the 3 inputs to 4 hidden nodes
# 4,1 is the 4 hidden nodes into the 1 output node
syn0 = 2 * np.random.random((3,4)) - 1
syn1 = 2 * np.random.random((4,1)) - 1

def predict():
    r,g,b = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    features = np.array([r,g,b])
    l0 = features
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    print("inputs ")
    print(r)
    print(g)
    print(b)
    print("font color should be ", l2)

# activation function
def sigmoid(x, d=False):
    if (d):
        return (x*(1-x))
    return 1/(1+np.exp(-x))

# trainign is basically creating an accurate representation of
# synapses / weights so the prediction can be close as possible
def train():
    global syn0
    global syn1
    # features = x = inputs
    features = np.array([
        [246,196,241],
        [69,84,235],
        [92,78,81],
        [131,20,46],
        [121,93,125],
        [216,100,18],
        [216,32,206],
        [45,76,198],
        [239,34,242],
        [228,94,49],
        [100,179,234],
        [178,152,240],
        [73,47,136],
        [182,69,215],
        [211,169,46],
        [176,34,25],
        [235,171,186],
        [63,114,45],
        [6,32,139],
        [156,57,27],
        [53,139,183],
        [168,90,100],
        [67,117,119],
        [142,99,55],
        [64,30,178],
        [223,88,140],
        [35,49,171],
        [125,135,210],
        [126,108,96],
        [210,111,8],
        [182,212,209],
        [214,231,161],
        [148,36,124],
        [233,44,246],
        [103,254,21],
        [199,183,64],
        [215,240,146],
        [123,211,159],
        [207,65,127],
        [217,27,114],
        [171,210,0],
        [14,114,240],
        [11,252,126],
        [180,206,145],
        [69,232,165],
        [61,172,170],
        [177,142,156],
        [117,22,46],
        [60,9,136],
        [126,36,56],
        [97,74,26],
        [113,111,93],
        [135,66,174],
        [168,76,131],
        [111,207,65],
        [227,173,199],
        [184,30,187],
        [98,222,67],
        [189,12,81],
        [93,135,127],
        [113,58,190],
        [6,252,241],
        [177,14,116],
        [71,228,18],
        [63,46,102],
        [127,83,208],
        [173,180,206],
        [173,208,247],
        [232,180,1],
        [122,64,156],
        [175,225,2],
        [72,156,106],
        [193,64,126],
        [204,177,69],
        [218,171,62],
        [139,42,199],
        [58,134,98],
        [30,167,120]
    ])

    # labels = y = outputs = "targets"
    labels = np.array([
        [0],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [0],
        [0],
        [1],
        [1],
        [0],
        [1],
        [0],
        [1],
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
        [0],
        [1],
        [1],
        [0],
        [0],
        [0],
        [0],
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
        [1],
        [1],
        [0],
        [0],
        [1],
        [0],
        [1],
        [1],
        [1],
        [0],
        [1],
        [0],
        [1],
        [1],
        [0],
        [0],
        [0],
        [1],
        [0],
        [1],
        [1],
        [0],
        [0],
        [1],
        [1],
        [1],
    ])

    # loop 60000 iterations so the weights have enough
    # times to adjust to lowering the error rate towards zero
    for j in range(60000):
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
        if (j % 5000) == 0:
            print ('Error Rate: ', str(np.mean(np.abs(l2_error))))

        # here we find the delta between our output layers
        # and the deriviate of the weights
        l2_delta = l2_error * sigmoid(l2, d=True)

        # we find the hidden layer deltas by taking the dot product of
        # the output deltas and output synapse weights
        l1_error = l2_delta.dot(syn1.T)

        # now we go back even further and use the hidden layer
        # erorr rates and find the hidden layer deltas
        l1_delta = l1_error * sigmoid(l1, d=True)

        # now that we have the delates between the expected and output
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

train()
print(predict())



'''
# inputs are R, G, B



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