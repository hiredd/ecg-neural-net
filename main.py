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
                skipped = 0 # lines skipped due to invalid characters
                reader = csv.reader(f)
                for row in reader:
                    try:
                        data.append([float(datum) for datum in row])
                    except ValueError:
                        skipped += 1
                if skipped > 0: 
                    logging.warning(str(skipped) + " lines were skipped.")
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
            sum_error += abs(row[-1]-self.feed_forward(row[:-1]))
        return sum_error/len(arr)

if __name__ == "__main__":
    print "-fold" in sys.argv
    if "-fold" not in sys.argv or sys.argv.index("-fold")==len(sys.argv)-1:
        # Separate training and test files
        training = load_csv("data/verify.csv")
        test = load_csv("data/test.csv")
        neural_net = NN_Network([3,5,1])
        neural_net.train(training)
        print "\nAverage error on test is: \033[91m" + str(float(neural_net.test(test)))
    else:
        try:
            fold_number = int(sys.argv[sys.argv.index("-fold")+1])
        except ValueError:
            raise SyntaxError("Syntax error in specifying number of folds.")
        data = load_csv("data/arrhythmia.csv")
        for fold in range(fold_number):
            neural_net = NN_Network([279,5,1])
            training1 = data[:fold*len(data)/fold_number]
            test = data[fold*len(data)/fold_number : (fold+1)*len(data)/fold_number]
            training2 = data[(fold+1)*len(data)/fold_number:]
            neural_net.train(training1)
            neural_net.train(training2)
            print "\nAverage error on fold #" + str(fold) + " test is: " + str(float(neural_net.test(test)))