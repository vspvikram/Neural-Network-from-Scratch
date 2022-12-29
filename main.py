# main file can be used to learn an input image using a neural network and 
# predict the same image by using just the input image coordinates

# overall module class which is subclassed by layer instances
# contains a forward method for computing the value for a given
# input and a backwards method for computing gradients using
# backpropogation.
import numpy as np
from tqdm import tqdm
import scipy.io as mat
from utils.modules import Module, Sigmoid, Linear
from utils.loss import MeanErrorLoss
from utils.optimizer import Adam
from utils.helper_functions import plot_image, plot_loss_curve
from utils.helper_functions import predict_image, test_fun, rmse



## overall neural network class
class Network(Module):
    def __init__(self, loss = None, optimizer=None):
        super(Network, self).__init__()
        # todo initializes layers, i.e. sigmoid, linear
        self.layers = {}
        
        self.loss = loss if loss else None
        
        self.optimizer = optimizer if optimizer else None

    # adding layer to the network
    def add(self, layer):
        self.layers[len(self.layers)+1] = layer
    
    # setting the loss layer to the network
    def set_loss(self, loss):
        self.loss = loss

    def set_optimizer_lr(self, learning_rate = 0.01, optimizer = None):
        self.optimizer = optimizer
        if self.optimizer:
            self.optimizer.set_config(self.layers)
            self.optimizer.learning_rate = learning_rate

    def forward(self, input):
        # todo compute forward pass through all initialized layers
        output = input
        for id, layer in self.layers.items():
            output = layer.forward(output)

        return output

    def backwards(self, grad):
        # todo iterate through layers and compute and store gradients
        grad_inputs = grad
        for idx, layer in reversed(self.layers.items()):
            grad_inputs = layer.backwards(grad_inputs)

    def update_weights(self, step_number):
        for idx in self.layers.keys():
            if self.layers[idx].is_trainable:
                self.optimizer.optimize(self.layers, idx, step_number) #layers, idx, step_number

    def predict(self, data, batch_size):
        # todo compute forward pass and output predictions
        for j in range(0, (data.shape[0] // batch_size)+1):
            if ((j+1)*batch_size // data.shape[0]) == 0:
                input_ = data[j*batch_size : (j+1)*batch_size,:]
            else:
                input_ = data[j*batch_size : (j+1)*batch_size,:]

            out = self.forward(input_)
            if j==0:
                output = out
            else:
                output = np.concatenate((output, out))

        return output

    def accuracy(self, test_data, test_labels):
        # todo evaluate accuracy of model on a test dataset
        pass



# function for training the network for a given number of iterations
def train(model, data, labels, num_iterations: int, minibatch_size: int, 
        learning_rate, schedule: list=None, gamma=0.1):
    '''Function to train the given model on the data in minibatches. The inputs to the function are:
    model: the model that is to be trained
    data: the input data in the format (#samples, #input_features)
    labels: the labels corresponding to the data in the format (#samples, #out_features)
    num_iterations: number of iterations to run during training
    minibatch_size: size of the minibatches to run in each iteration
    learning_rate: the initial learning rate
    schedule: a list with the schedule to update the learning rate after fixed steps
    gamma: a multiplier to update the learning rate when schedule step is reached
    
    Output
    model: trained model
    losses: list of losses at each iteration'''
    
    adam = Adam()
    model.set_optimizer_lr(learning_rate, optimizer = adam)

    losses = []
    iteration = 0
    if schedule:
        curr_schedule = 0
    
    batches = data.shape[0] // minibatch_size
    if data.shape[0] % (minibatch_size * batches) != 0:
        batches += 1

    pbar = tqdm(total = num_iterations)
    while iteration < num_iterations:
        # for j in tqdm(range(0, batches), desc="Training",
        #             position=0, leave=False):
        for j in range(0, batches):
            if (iteration > num_iterations - 1):
                break
            iteration += 1
            if ((j+1)*minibatch_size // data.shape[0]) == 0:
                X = data[j*minibatch_size : (j+1)*minibatch_size,:]
                y = labels[j*minibatch_size : (j+1)*minibatch_size,:]
            else:
                X = data[j*minibatch_size : ,:]
                y = labels[j*minibatch_size : ,:]

            y_hat = model.forward(X)

            loss = np.mean(model.loss.forward(y_hat, y))
            losses.append(loss)

            gradient = model.loss.backwards()

            model.backwards(gradient)

            model.update_weights(step_number=iteration) #step_number=j+1

            if schedule and (iteration == schedule[curr_schedule]):
                model.optimizer.learning_rate = model.optimizer.learning_rate * gamma
                curr_schedule = min(len(schedule)-1, curr_schedule+1)


        # epoch_loss = sum(epoch_loss)/len(epoch_loss)
        # losses.append(epoch_loss)
            if iteration % (num_iterations // 10) ==0:
                pbar.update((num_iterations // 10))
                print("Iteration: {}/{} - Loss: {:.4f}".format(iteration, num_iterations, loss))
    
    pbar.close()
    return model, losses



# creating and training the model with the given hyperparameters
def create_and_train_model(x, y, hyperparameters, layers, name):
    '''Creates and train the model with given layer structure
    x: input x with shape (#samples, #input_features)
    y: labels without normalized pixel values, shape (#samples, #output_dimentions)
    hyperparamets: dictionary of the hyperparameters
    layers: list of the hidden units in each layer
    name: name of the model to give'''
    assert layers[0] == x.shape[1], "Input shape and the model structure does not match"
    assert layers[-1] == y.shape[1], "Label dimensions and the model structure does not match"
    X_train_orig = x
    y_train_orig = y/255 #np.reshape(y1, (y1.size))
    # randomly shuffling the data
    p = np.random.permutation(len(X_train_orig))
    X_train = X_train_orig[p]
    y_train = y_train_orig[p]

    learning_rate = hyperparameters["learning_rate"]
    schedule = hyperparameters["schedule"]
    batch_size = hyperparameters["batch_size"]
    num_iterations = hyperparameters["num_iterations"]

    # building model
    model = Network()
    for i in range(len(layers)-2):
        model.add(Linear(layers[i], layers[i+1]))
        model.add(Sigmoid())
    model.add(Linear(layers[-2], layers[-1]))
    
    loss = MeanErrorLoss()
    model.set_loss(loss)
    plot_image(x, y_train_orig, name=f"{name}_orig_image.png")
    y_hat_prev = model.predict(X_train, batch_size=batch_size)
    print("The RMSE of the {} results before training: {}".format(name, rmse(y_train, y_hat_prev)))

    model, losses = train(model, X_train, y_train, num_iterations=num_iterations, 
            minibatch_size=batch_size, learning_rate=learning_rate, schedule=schedule)#schedule=schedule
    plot_loss_curve(losses, name)
    return model
    # y_hat = model.predict(X_train_orig, batch_size=batch_size)
    
    # print("The RMSE of the {} results: {}".format(name, rmse(y_train_orig, y_hat)))
    
    # y_hat = (y_hat * 255).astype(np.int16)
    # plot_image(x1, y_hat, name=f"{name}_pred_image.png")


# main code starts here
if __name__ == "__main__":
    # input image files 
    mat_file = mat.loadmat("nn_data.mat")

    # first image
    # Note: This is just one method to import images. User can use other method to import individual image files.
    x1 = mat_file["X1"].astype(np.int32)
    y1 = mat_file["Y1"].astype(np.int32)

    # Hyperparameters
    hyperparameters_x1 = {"learning_rate": 0.001,
        "schedule": [50000, 70000, 80000],
        "batch_size": 4,
        "num_iterations": 100000}
    layers = [2, 512, 512, 512, 1]

    # building the model and predicting the image (saved in the same folder)
    x1_model = create_and_train_model(x1, y1, hyperparameters_x1, layers=layers, name="X1")
    predict_image(x1_model, x1, y1/255, hyperparameters_x1["batch_size"], name="X1")


    # # second image 
    # x2 = mat_file["X2"].astype(np.int32)
    # y2 = mat_file["Y2"].astype(np.int32)

    # hyperparameters_x2 = {"learning_rate": 0.001,
    #     "schedule": [40000, 60000, 80000],
    #     "batch_size": 12,
    #     "num_iterations": 200000}
    # layers_x2 = [2, 300, 512, 300, 128, 3]

    # # building the model and predicting the image (saved in the same folder)
    # x2_model = create_and_train_model(x2, y2, hyperparameters_x2, layers=layers_x2, name="X2")
    # predict_image(x2_model, x2, y2/255, hyperparameters_x2["batch_size"], name="X2")


    # print("hi")