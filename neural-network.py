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

        # Link weight matrices
        # wih: Weights for Input->Hidden
        # who: Weights for Hidden->Output
        # Weights inside the matrices are w_i_j, where link is from node i to node j in the next layer
        # w11 w12
        # w21 w22 etc
        self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)

    # Refine the weights in the network after using a training set
    def train():
        pass

    # Ask the network for output based on input
    def query():
        pass