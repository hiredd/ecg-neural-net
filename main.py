import csv
import logging
import numpy as np
import sys

LEARNING_RATE = 0.05

# Helper functions for output squashing and backprop
def tanh(in_array):
    """Computes the tanh function for a numpy array"""
    return np.tanh(in_array)

def tanh_(in_array):
    """Computes the derivative of the tanh function for a numpy array"""
    return 1-np.power(tanh(in_array),2)

def load_csv(file_name):
    """Loads a csv file as a 2-dimensional array for input"""
    data = []
    if ".csv" in file_name:
        try:
            with open(file_name) as f:
                reader = csv.reader(f)
                for row in reader:
                    data.append([float(datum) for datum in row])
            return np.array(data)       
        except IOError:
            logging.error("Error occurred while reading file.")
            sys.exit(1)
    else:
        logging.error("Incorrect file format uploaded.")
        sys.exit(1)


class NN_Layer:

    def __init__(self, in_size, out_size):
        # Initialize trainable parameters
        self.weights = np.matrix(np.random.randn(in_size, out_size))
        #self.bias = np.random.randn(out_size)
        self.bias = np.zeros(out_size)
        self.input = None # for convenience, store input to this layer
        self.output = None # for convenience, store output from this layer

    def feed_forward(self, arr):
        """Computes the output from a layer of nodes given an np array of size in_size"""
        self.input = arr
        raw_sum = np.dot(arr, self.weights) + self.bias
        self.output = tanh(raw_sum)
        return self.output

class NN_Network:

    def __init__(self, sizes):
        # Initialize NN_Layer objects according to supplied sizes
        self.layers = [NN_Layer(sizes[i], sizes[i+1]) for i in np.arange(len(sizes)-1)]

    def feed_forward(self, arr):
        """Computes the forward pass of a neural network given an input, and returns an np array"""
        for layer in self.layers:
            arr = layer.feed_forward(arr)
        return arr

    def backprop(self,output,expected):
        """Performs backpropagation through the neural network for one entry, using squared error"""
        error = expected-output
        for layer in self.layers[::-1]:
            delta = np.multiply(error, tanh_(layer.output))
            error = np.dot(delta, layer.weights.T)
            layer.weights += np.dot(layer.input.T, delta)

    def train(self, arr):
        """Takes a 2D array of floats as input, formatted with last item as labels and all others as attributes"""
        for _ in range(50):
            for row in arr:
                forward = self.feed_forward(np.matrix(row[:-1]))
                self.backprop(forward,row[-1])

    def test(self, arr):
        sum_error = 0
        for row in arr:
            print "\033[93m"+str(float(self.feed_forward(row[:-1])))+", "+str(row[-1])
            sum_error += abs(row[-1]-self.feed_forward(row[:-1]))
        return sum_error/len(arr)

if __name__ == "__main__":
    training = load_csv("data/verify.csv")
    neural_net = NN_Network([3,5,1]) # has two layers of size 4 and 1, accepts input of size 3
    neural_net.train(training)
    test = load_csv("data/test.csv")
    print "\nAverage error on test is: " + str(float(neural_net.test(test)))
