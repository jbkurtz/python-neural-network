import numpy
import scipy.special

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
    def train(self, inputs_list, targets_list):
        # Convert inputs list into 2D Array
        inputs = numpy.array(inputs_list, ndmin=2) .T

        # Calculate signal into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Calculate signal emerging from hidden layer
        hidden_outputs = self.activation(hidden_inputs)

        # Calculate signal into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Calculate signal emerging from final output layer
        final_outputs = self.activation(final_inputs)

        # Convert targets list into 2D Array
        targets = numpy.array(targets_list, ndmin=2) .T

        # Calculate error
        output_errors = targets - final_outputs

        # Calculate hidden error
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # Update weights for links between hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)), numpy.transpose(hidden_outputs))

        # Update the weights for links between input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))


    # Activation function, currently a sigmoid function
    def activation(self, value):
        return scipy.special.expit(value)

    # Ask the network for output based on input
    def query(self, inputs_list):
        # Convert inputs list into 2D Array
        inputs = numpy.array(inputs_list, ndmin=2) .T

        # Calculate signal into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Calculate signal emerging from hidden layer
        hidden_outputs = self.activation(hidden_inputs)

        # Calculate signal into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Calculate signal emerging from final output layer
        final_outputs = self.activation(final_inputs)

        return final_outputs