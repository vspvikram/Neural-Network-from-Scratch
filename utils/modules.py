import numpy as np


class Module():
    def __init__(self, learning_rate=None):
        self.prev = None # previous network (linked list of layers)
        self.output = None # output of forward call for backprop.
        
        self.learning_rate = 1E-2 # class-level learning rate

    def __call__(self, input):
        if isinstance(input, Module):
            raise NotImplementedError
            # todo. chain two networks together with module1(module2(x))
            # update prev and output
        else:
            raise NotImplementedError
            # todo. evaluate on an input.
            # update output

        return self

    def forward(self, *input):
        raise NotImplementedError

    def backwards(self, *input):
        raise NotImplementedError


# sigmoid non-linearity
class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.is_trainable = False

    def forward(self, input):
        # todo. compute sigmoid, update fields
        self.input = input
        self.output = self.sigmoid(input)
        return self.output

    def sigmoid(self, input):
        return 1/(1 + np.exp(-1 * input))
        

    def backwards(self, gradient):
        # todo. compute gradients with backpropogation and data from forward pass
        # gradient should have the shape (batch_size, output_size)
        self.sig_deriv = gradient * (self.output * (1 - self.output))
        return self.sig_deriv


# linear (i.e. linear transformation) layer
class Linear(Module):
    '''Linear NN layer with initialization parameters
    input_size: size of the input to the layer
    output_size: size of the output given by this layer'''
    def __init__(self, input_size, output_size, is_input=False, learning_rate=None):
        super().__init__(learning_rate=learning_rate)
        # todo. initialize weights and biases. 
        self.weights = np.random.rand(input_size, output_size) - 0.5 # 0.5 to make the range between -0.5 and 0.5
        self.biases = np.random.rand(1, output_size) - 0.5 # to make the values lie between -0.5 and 0.5
        self.is_trainable = True

    def forward(self, input):  # input has shape (batch_size, input_size)
        # todo compute forward pass through linear input
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.biases # shape (batch_size, output_size)

        return self.output

    def backwards(self, gradient):
        # todo compute and store gradients using backpropogation
        # gradient should have shape of (batch_size, output_size)
        self.grad_weights = np.dot(self.input.T, gradient)/self.input.shape[0] # shape (input_size, output_size)
        grad_inputs = np.dot(gradient, self.weights.T) # shape (batch_size, input_size)
        self.grad_biases = gradient.mean(axis=0)

        # updating the weights and biases
        # self.weights = self.weights - self.learning_rate * grad_weights
        # self.biases = (self.biases - self.learning_rate * gradient).mean(axis=0)

        return grad_inputs
