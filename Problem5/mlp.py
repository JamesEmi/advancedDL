"""
Spring 2024, 10-707
Homework 1
Problem 5: MLP
TAs in charge: 
    Jiatai Li (jiatail)
    Kaiwen Geng (kgeng)
    Torin Kovach (tkovach)

IMPORTANT:
    DO NOT change any function signatures

Jan 2024
"""

import numpy as np


def random_weight_init(indim, outdim):
    b = np.sqrt(6) / np.sqrt(indim + outdim)
    return np.random.uniform(-b, b, (indim, outdim))


def zeros_bias_init(outdim):
    return np.zeros((outdim, 1))


class Transform:
    """
    This is the base class. You do not need to change anything.

    Please read the comments in this class carefully.
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
        In this function, we accumulate the gradient values instead of assigning
        the gradient values. This allows us to call forward and backward multiple
        times while only update parameters once.
        Compute and save the gradients wrt the parameters for step()
        Return grad_wrt_x which will be the grad_wrt_out for previous Transform
        """
        pass

    def step(self):
        """
        Apply gradients to update the parameters
        """
        pass

    def zerograd(self):
        """
        This is used to Reset the gradients.
        Usually called before backward()
        """
        pass


class ReLU(Transform):
    """
    Implement this class
    """

    def __init__(self):
        Transform.__init__(self)
        pass

    def forward(self, x, train=True):
        pass

    def backward(self, grad_wrt_out):
        pass


class LinearMap(Transform):
    """
    Implement this class
    Please use *_init() functions given at the beginning of this file
    """

    def __init__(self, indim, outdim, alpha=0.1, lr=0.01):
        Transform.__init__(self)
        """
        indim: input dimension
        outdim: output dimension
        alpha: parameter for momentum updates
        lr: learning rate
        """
        self.alpha = alpha
        self.lr = lr
        self.W = random_weight_init(indim, outdim)
        self.b = zeros_bias_init(outdim)

    def forward(self, x):
        """
        x shape (batch_size, indim)
        return shape (batch_size, outdim)
        """
        pass

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (batch_size, outdim)
        return shape (batch_size, indim)
        Your backward call should Accumulate gradients.
        """
        pass

    def step(self):
        """
        apply gradients calculated by backward() to update the parameters

        Make sure your gradient step takes into account momentum.
        Use alpha as the momentum parameter.
        """
        pass

    def zerograd(self):
        # reset parameters
        pass

    def getW(self):
        # return weights
        return self.W

    def getb(self):
        # return bias
        return self.b

    def loadparams(self, w, b):
        # Used for Autograder. Do not change.
        self.W, self.b = w, b


class SoftmaxCrossEntropyLoss:
    """
    Implement this class
    """

    def forward(self, logits, labels):
        """
        logits are pre-softmax scores, labels are true labels of given inputs
        labels are one-hot encoded
        logits and labels are in the shape of (batch_size, num_classes)
        returns loss as scalar
        (your loss should be a mean value on batch_size)
        """
        pass

    def backward(self):
        """
        return shape (batch_size, num_classes)
        (don't forget to divide by batch_size because your loss is a mean)
        """
        pass

    def getAccu(self):
        """
        return accuracy here (as you wish)
        This part is not autograded.
        """
        pass


class SingleLayerMLP(Transform):
    """
    Implement this class
    """

    def __init__(self, indim, outdim, hiddenlayer=100, alpha=0.1, lr=0.01):
        Transform.__init__(self)
        pass

    def forward(self, x, train=True):
        """
        x shape (batch_size, indim)
        return shape (batch_size, outdim)
        """
        pass

    def backward(self, grad_wrt_out):
        pass

    def step(self):
        pass

    def zerograd(self):
        pass

    def loadparams(self, Ws, bs):
        """
        use LinearMap.loadparams() to implement this
        Ws is a list, whose element is weights array of a layer, first layer first
        bs for bias similarly
        e.g., Ws may be [LinearMap1.W, LinearMap2.W]
        Used for autograder.
        """
        pass

    def getWs(self):
        """
        Return the weights for each layer
        You need to implement this.
        Return weights for first layer then second and so on...
        """
        pass

    def getbs(self):
        """
        Return the biases for each layer
        You need to implement this.
        Return bias for first layer then second and so on...
        """
        pass


class TwoLayerMLP(Transform):
    """
    Implement this class
    Everything similar to SingleLayerMLP
    """

    def __init__(self, inp, outp, hiddenlayers=[100, 100], alpha=0.1, lr=0.01):
        Transform.__init__(self)
        pass

    def forward(self, x, train=True):
        pass

    def backward(self, grad_wrt_out):
        pass

    def step(self):
        pass

    def zerograd(self):
        pass

    def loadparams(self, Ws, bs):
        pass

    def getWs(self):
        pass

    def getbs(self):
        pass


class Dropout(Transform):
    """
    Implement this class
    """

    def __init__(self, p=0.5):
        Transform.__init__(self)
        """
        p is the Dropout probability
        """
        pass

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, train=True):
        """
        Get and apply a mask generated from np.random.binomial during training
        Scale your output accordingly during testing
        """
        pass

    def backward(self, grad_wrt_out):
        """
        This method is only called during trianing.
        """
        pass


class BatchNorm(Transform):
    """
    Implement this class
    """

    def __init__(self, indim, alpha=0.9, lr=0.01, mm=0.9):
        Transform.__init__(self)
        """
        You shouldn't need to edit anything in init
        """
        self.alpha = alpha  # parameter for running average of mean and variance
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None
        self.lr = lr
        self.mm = mm  # parameter for updating gamma and beta
        """
        The following attributes will be tested
        """
        self.var = np.ones((1, indim))
        self.mean = np.zeros((1, indim))

        self.gamma = np.ones((1, indim))
        self.beta = np.zeros((1, indim))

        """
        gradient parameters
        """
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

        """
        momentum parameters
        """
        self.mgamma = np.zeros_like(self.gamma)
        self.mbeta = np.zeros_like(self.beta)

        """
        inference parameters
        """
        self.running_mean = np.zeros((1, indim))
        self.running_var = np.ones((1, indim))

    def __call__(self, x, train=True):
        return self.forward(x, train)

    def forward(self, x, train=True):
        """
        x shape (batch_size, indim)
        return shape (batch_size, indim)
        Please use batch mean and variance to update running averages during training,
        and use the running averages to normalize input during testing.
        """
        pass

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (batch_size, indim)
        return shape (batch_size, indim)
        """
        pass

    def step(self):
        """
        apply gradients calculated by backward() to update the parameters
        Make sure your gradient step takes into account momentum.
        Use mm as the momentum parameter.
        """
        pass

    def zerograd(self):
        # reset parameters
        pass

    def getgamma(self):
        # return gamma
        return self.gamma

    def getbeta(self):
        # return beta
        return self.beta

    def loadparams(self, gamma, beta):
        # Used for Autograder. Do not change.
        self.gamma, self.beta = gamma, beta


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
    trainX = data["trainX"] / 255.0
    trainy = labels2onehot(data["trainy"])
    testX = data["testX"] / 255.0
    testy = labels2onehot(data["testy"])
