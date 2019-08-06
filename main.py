import numpy
import matplotlib.pyplot
import neuralnetwork as nn

# Number of nodes
# Number of input pixels
input_nodes = 784
# Arbitrary 
hidden_nodes = 100
# Number of options for output
output_nodes = 10 

learning_rate = 0.3

network = nn.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Train the network
with open('mnist_dataset/mnist_train_100.csv') as fp:
    training_data = fp.readlines()

for record in training_data:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.1
    targets = numpy.zeros(output_nodes) + 0.1
    targets[int(all_values[0])] = 0.99
    network.train(inputs, targets)

# Test the network
with open('mnist_dataset/mnist_test_10.csv') as fp:
    test_data = fp.readlines()

scorecard = []

for record in test_data:
    all_values = record.split(',')
    print('Correct label: ', all_values[0])
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.1
    outputs = network.query(inputs)
    print("Network's answer: ", numpy.argmax(outputs))
    if( int(all_values[0]) == numpy.argmax(outputs) ):
        scorecard.append(1)
    else:
        scorecard.append(0)

print('Scorecard: ', scorecard)

scorecard_array = numpy.asarray(scorecard)

print('Performance: ', scorecard_array.sum() / scorecard_array.size)