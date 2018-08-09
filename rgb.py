import sys
import numpy as np

# activation function
def sigmoid(x, d=False):
    if (d):
        return (x*(1-x))
    return 1/(1+np.exp(-x))

# used to train!
def sqrErr(t, o):
    return .5 * ((t-o) ** 2)

def train():
    # features = x = inputs
    features = []

    # labels = y = outputs = "targets"
    labels = []

    # weights / synapses / connections
    # syn0 is the
    syn0 = 2 * np.random.random((3,4)) - 1
    syn1 = 2 * np.random.random((4,1)) - 1
    print syn0

np.random.seed(2)
learning_rate = 0.5

# inputs are R, G, B
r,g,b = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
inputs = [r,g,b]

train()




'''

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