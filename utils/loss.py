import numpy as np


# generic loss layer for loss functions
class Loss:
    def __init__(self):
        self.prev = None

    def __call__(self, input):
        self.prev = input
        return self

    def forward(self, input, labels):
        raise NotImplementedError

    def backwards(self):
        raise NotImplementedError


# MSE loss function
class MeanErrorLoss(Loss):
    def __init__(self):
        super(MeanErrorLoss, self).__init__()

    def forward(self, input, labels):  # input has shape (batch_size, input_size)
        # todo compute loss, update fields
        self.prev = input
        self.labels = labels
        self.loss = np.mean(np.power(labels - input, 2), axis=1)
        return self.loss

    def backwards(self):
        # todo compute gradient using backpropogation
        return 2 * (self.prev - self.labels)/self.labels.shape[1]