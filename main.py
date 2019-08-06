import numpy
import matplotlib.pyplot
import neuralnetwork as nn

def train(network, output_nodes):
    # Train the network
    with open('mnist_dataset/mnist_train.csv') as fp:
        training_data = fp.readlines()

    for record in training_data:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.1
        targets = numpy.zeros(output_nodes) + 0.1
        targets[int(all_values[0])] = 0.99
        network.train(inputs, targets)

def test_all(network):
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

    print('Performance: ', scorecard_array.sum() / scorecard_array.size * 100.0, "%\n")

def test(network, img):
    with open('mnist_dataset/mnist_test_10.csv') as fp:
        test_data = fp.readlines()

    all_values = test_data[img].split(',')
    print('Correct label: ', all_values[0])
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.1
    outputs = network.query(inputs)
    print("Network's answer: ", numpy.argmax(outputs), "\n")

def main():
    # Number of nodes
    # Number of input pixels
    input_nodes = 784
    # Arbitrary 
    hidden_nodes = 100
    # Number of options for output
    output_nodes = 10 

    learning_rate = 0.3

    network = nn.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    print("Training network! May take some time...")
    train(network, output_nodes)

    print("Test the network!")
    print("Options: ")
    print("    Enter a number, 0-9, to test the image from mnist_dataset/images/test")
    print("    Enter 'all' to test against all images in mnist_dataset/images/test")
    print("    Type 'q' to quit")

    choice = input("Input: ")

    while choice is not 'q':
        try:
            input_num = int(choice)
            test(network, input_num)
        except ValueError:
            if choice == 'all':
                test_all(network)
            else:
                print("Error - unknown command")

        choice = input("Input: ")

main()