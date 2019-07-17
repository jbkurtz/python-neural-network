import numpy

# Base Neural Network Class
class neuralNetwork:

    # Initialize the number of input, hidden, and output nodes in the network
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        # Set number of nodes in each layer
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes

        # Set learning rate
        self.lr = learningRate
        pass

    # Refine the weights in the network after using a training set
    def train():
        pass

    # Ask the network for output based on input
    def query():
        pass