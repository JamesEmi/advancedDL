"""
Spring 2024, 10-707
Homework 1
Problem 6: CNN
TAs in charge: 
    Jiatai Li (jiatail)
    Kaiwen Geng (kgeng)
    Torin Kovach (tkovach)

IMPORTANT:
    DO NOT change any function signatures

    Some modules in Problem 6 like ReLU and LinearLayer are similar to Problem 5
    but not exactly same. Read their commented instructions carefully.

Jan 2024
"""

import numpy as np


def im2col(X, k_height, k_width, padding=1, stride=1):
    """
    Construct the im2col matrix of intput feature map X.
    X: 4D tensor of shape [N, C, H, W], input feature map
    k_height, k_width: height and width of convolution kernel
    return a 2D array of shape (C*k_height*k_width, H*W*N)
    The axes ordering need to be (C, k_height, k_width, H, W, N) here, while in
    reality it can be other ways if it weren't for autograding tests.
    """
    pass


def im2col_bw(grad_X_col, X_shape, k_height, k_width, padding=1, stride=1):
    """
    Map gradient w.r.t. im2col output back to the feature map.
    grad_X_col: a 2D array
    return X_grad as a 4D array in X_shape
    """
    pass


class Transform:
    """
    This is the base class. You do not need to change anything.
    Read the comments in this class carefully.
    """

    def __init__(self):
        """
        Initialize any parameters
        """
        pass

    def forward(self, x):
        """
        x should be passed as column vectors
        """
        pass

    def backward(self, grad_wrt_out):
        """
        Unlike Problem 5 MLP, here we no longer accumulate the gradient values,
        we assign new gradients directly. This means we should call update()
        every time we do forward and backward, which is fine. Consequently, in
        Problem 6 zerograd() is not needed any more.
        Compute and save the gradients wrt the parameters for update()
        Read comments in each class to see what to return.
        """
        pass

    def update(self, learning_rate, momentum_coeff):
        """
        Apply gradients to update the parameters
        """
        pass


class ReLU(Transform):
    """
    Implement this class
    """

    def __init__(self):
        Transform.__init__(self)

    def forward(self, x, train=True):
        """
        returns ReLU(x)
        """
        pass

    def backward(self, dloss):
        """
        dloss is the gradients wrt the output of ReLU
        returns gradients wrt the input to ReLU
        """
        pass


class Flatten(Transform):
    """
    Implement this class
    """

    def forward(self, x):
        """
        returns Flatten(x)
        """
        pass

    def backward(self, dloss):
        """
        dLoss is the gradients wrt the output of Flatten
        returns gradients wrt the input to Flatten
        """
        pass


class Conv(Transform):
    """
    Implement this class - Convolution Layer
    """

    def __init__(self, input_shape, filter_shape, rand_seed=None):
        """
        input_shape is a tuple: (channels, height, width)
        filter_shape is a tuple: (num of filters, filter height, filter width)
        weights shape (number of filters, number of input channels, filter height, filter width)
        Use Xavier initialization for weights, as instructed on handout
        Initialze biases as an array of zeros in shape of (num of filters, 1)
        """
        if rand_seed is not None:
            np.random.seed(rand_seed)
        self.C, self.H, self.Width = input_shape
        self.num_filters, self.k_height, self.k_width = filter_shape
        b = np.sqrt(6) / np.sqrt(
            (self.C + self.num_filters) * self.k_height * self.k_width
        )
        self.W = np.random.uniform(
            -b, b, (self.num_filters, self.C, self.k_height, self.k_width)
        )
        self.b = np.zeros((self.num_filters, 1))

    def forward(self, inputs, stride=1, pad=2):
        """
        Forward pass of convolution between input and filters
        inputs is in the shape of (batch_size, num of channels, height, width)
        Return the output of convolution operation in shape (batch_size, num of filters, height, width)
        use im2col here to vectorize your computations
        """
        pass

    def backward(self, dloss):
        """
        Read Transform.backward()'s docstring in this file
        dloss shape (batch_size, num of filters, output height, output width)
        Return [gradient wrt weights, gradient wrt biases, gradient wrt input to this layer]
        use im2col_bw here to vectorize your computations
        """
        pass

    def update(self, learning_rate=0.001, momentum_coeff=0.5):
        """
        Update weights and biases with gradients calculated by backward()
        Use the same momentum formula as in Problem 5.
        """
        pass

    def get_wb_conv(self):
        """
        Return weights and biases
        """
        return self.W, self.b


class MaxPool(Transform):
    """
    Implement this class - MaxPool layer
    """

    def __init__(self, filter_shape, stride):
        """
        filter_shape is (filter_height, filter_width)
        stride is a scalar
        """
        pass

    def forward(self, inputs):
        """
        forward pass of MaxPool
        inputs: (N, C, H, W)
        """
        pass

    def backward(self, dloss):
        """
        dloss is the gradients wrt the output of forward()
        """
        pass


class LinearLayer(Transform):
    """
    Implement this class - Linear layer
    """

    def __init__(self, indim, outdim, rand_seed=None):
        """
        indim, outdim: input and output dimensions
        weights shape (indim,outdim)
        Use Xavier initialization for weights, as instructed on handout
        Initialze biases as an array of ones in shape of (outdim,1)
        """
        if rand_seed is not None:
            np.random.seed(rand_seed)
        b = np.sqrt(6) / np.sqrt(indim + outdim)
        self.W = np.random.uniform(-b, b, (indim, outdim))
        self.b = np.zeros((outdim, 1))

    def forward(self, inputs):
        """
        Forward pass of linear layer
        inputs shape (batch_size, indim)
        """
        pass

    def backward(self, dloss):
        """
        Read Transform.backward()'s docstring in this file
        dloss shape (batch_size, outdim)
        Return [gradient wrt weights, gradient wrt biases, gradient wrt input to this layer]
        """
        pass

    def update(self, learning_rate=0.001, momentum_coeff=0.5):
        """
        Similar to Conv.update()
        """
        pass

    def get_wb_fc(self):
        """
        Return weights and biases
        """
        return self.W, self.b


class SoftMaxCrossEntropyLoss:
    """
    Implement this class
    """

    def forward(self, logits, labels, get_predictions=False):
        """
        logits are pre-softmax scores, labels are true labels of given inputs
        labels are one-hot encoded
        logits and labels are in the shape of (batch_size, num_classes)
        returns loss as scalar
        (your loss should be the mean loss over the batch)
        """
        pass

    def backward(self):
        """
        return shape (batch_size, num_classes)
        Remeber to divide by batch_size so the gradients correspond to the mean loss
        """
        pass

    def getAccu(self):
        """
        Implement as you wish, not autograded.
        """
        pass


class ConvNet:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool -> Linear -> Softmax
    For the above network run forward, backward and update
    """

    def __init__(self, rand_seed=None):
        """
        Initialize Conv, ReLU, MaxPool, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 1x6x6
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then initialize linear layer with output 10 neurons
        Initialize SoftMaxCrossEntropy object.
        Remember to pass in the rand_seed to initialize all layers,
        otherwise you may not pass autograder.
        """
        pass

    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => true labels of shape (batch_size, num_classes)
        Return loss and predicted labels after one forward pass
        """
        pass

    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        pass

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        pass


class ConvNetTwo:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool -> Linear -> Softmax
    For the above network run forward, backward and update
    """

    def __init__(self):
        """
        Initialize Conv, ReLU, MaxPool, Conv, ReLU,LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 5x4x4
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then initialize linear layer with output 10 neurons
        Initialize SotMaxCrossEntropy object
        """
        pass

    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape (batch, channels, height, width)
        2. labels => true labels of shape (batch_size, num_classes)
        Return loss and predicted labels after one forward pass
        """
        pass

    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        pass

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        pass


class ConvNetThree:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool -> Conv -> Relu -> Conv -> Relu -> Linear -> Softmax
    For the above network run forward, backward and update
    """

    def __init__(self):
        """
        Initialize Conv, ReLU, MaxPool, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 5x4x4
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then Conv with filter size of 5x4x4
        then apply Relu
        then Conv with filter size of 5x4x4
        then apply Relu
        then initialize linear layer with output 10 neurons
        Initialize SotMaxCrossEntropy object
        """
        pass

    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => true labels of shape (batch_size, num_classes)
        Return loss and predicted labels after one forward pass
        """
        pass

    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        pass

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        pass


def labels2onehot(labels):
    return np.eye(np.max(labels) + 1)[labels].astype(np.float32)


if __name__ == "__main__":
    """
    You can implement your training and testing loop here.
    You MUST use your class implementations to train the model and to get the results.
    DO NOT use pytorch or tensorflow get the results. The results generated using these
    libraries will be different as compared to your implementation.
    """
    import pickle

    # change this to where you downloaded the file,
    # usually ends with 'cifar10-subset.pkl'
    CIFAR_FILENAME = "../cifar10-subset.pkl"
    with open(CIFAR_FILENAME, "rb") as f:
        data = pickle.load(f)

    # preprocess
    trainX = data["trainX"].reshape(-1, 3, 32, 32) / 255.0
    trainy = labels2onehot(data["trainy"])
    testX = data["testX"].reshape(-1, 3, 32, 32) / 255.0
    testy = labels2onehot(data["testy"])
