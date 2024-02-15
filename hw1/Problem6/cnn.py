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
    return a 2D array of shape (C*k_height*k_width, H_out*W_out*N)
    The axes ordering need to be (C, k_height, k_width, H, W, N) here, while in
    reality it can be other ways if it weren't for autograding tests.
    """
    N, C, H, W = X.shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    # X_padded = np.pad(X, pad_width=1)
    X_padded = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    H_out = (H_padded - k_height) // stride + 1
    W_out = (W_padded - k_width) // stride + 1

    i0 = np.repeat(np.arange(k_height), k_width)
    i0 = np.tile(i0,C)
    i1 = stride * np.repeat(np.arange(H_out), W_out)
    j0 = np.tile(np.arange(k_width), k_height*C)
    j1 = stride * np.tile(np.arange(W_out), H_out)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)  #how does this work?
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    
    
    
    k = np.repeat(np.arange(C), k_height * k_width).reshape(-1, 1)
    
    # Ensure k is broadcastable with i and j
    # If i and j index spatial locations, k needs to align with those indexes correctly
    # One approach is to add a new axis to i and j to make their shapes compatible with k for broadcasting
    # i = i.reshape(1, *i.shape)
    # j = j.reshape(1, *j.shape)
    patches = X_padded[:, k, i, j]
    C = X.shape[1]
    cols = patches.transpose(1, 2, 0).reshape(k_height * k_width * C, -1)
    return cols

def im2col_bw(grad_X_col, X_shape, k_height, k_width, padding=1, stride=1):
    """
    Map gradient w.r.t. im2col output back to the feature map.
    grad_X_col: a 2D array
    return X_grad as a 4D array in X_shape
    """
    N, C, H, W = X_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    H_out = (H_padded - k_height) // stride + 1
    W_out = (W_padded - k_width) // stride + 1

    X_grad_padded = np.zeros((N, C, H_padded, W_padded))

    i0 = np.repeat(np.arange(k_height), k_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(H_out), W_out)
    j0 = np.tile(np.arange(k_width), k_height * C)
    j1 = stride * np.tile(np.arange(H_out), W_out)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), k_height * k_width).reshape(-1, 1)

    # Reverse operation: add gradients instead of indexing
    np.add.at(X_grad_padded, (slice(None), k, i, j), grad_X_col.reshape(C*k_height*k_width, -1, N).transpose(2,0,1))

    # Remove padding
    if padding > 0:
        return X_grad_padded[:, :, padding:-padding, padding:-padding]
    else:
        return X_grad_padded

  
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
        self.mask = (x>0).astype(x.dtype) #save the mask of positives
        return x*self.mask

    def backward(self, dloss):
        """
        dloss is the gradients wrt the output of ReLU
        returns gradients wrt the input to ReLU
        """
        return dloss * self.mask


class Flatten(Transform):
    """
    Implement this class
    """
    
    def __init__(self):
        self.original_shape = None  # To store the shape of the input tensor

    def forward(self, x):
        """
        returns Flatten(x)
        """
        self.original_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dloss):
        """
        dLoss is the gradients wrt the output of Flatten
        returns gradients wrt the input to Flatten
        """
        return dloss.reshape(self.original_shape)


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
        self.stride = 1
        self.pad = 1
        
        # Initialize momentum velocity terms for weights and biases
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

    def forward(self, inputs, stride=1, pad=2):
        """
        Forward pass of convolution between input and filters
        inputs is in the shape of (batch_size, num of channels, height, width)
        Return the output of convolution operation in shape (batch_size, num of filters, height, width)
        use im2col here to vectorize your computations
        """
        self.input_cols = im2col(inputs, self.k_height, self.k_width, pad, stride)
        kernel_reshaped = self.W.reshape((self.num_filters, self.k_height*self.k_width*self.C))
        conv_output = np.dot(kernel_reshaped, self.input_cols) + self.b
        
        H_padded, W_padded = self.H + 2 * pad, self.Width + 2 * pad

        H_out = (H_padded - self.k_height) // stride + 1
        W_out = (W_padded - self.k_width) // stride + 1
        
        batch_size = inputs.shape[0]
        
        conv_output = conv_output.reshape(self.num_filters, H_out, W_out, batch_size)
        conv_output = conv_output.transpose(3,0,1,2)
        
        return conv_output 
        

    def backward(self, dloss):
        """
        Read Transform.backward()'s docstring in this file
        dloss shape (batch_size, num of filters, output height, output width)
        Return [gradient wrt weights, gradient wrt biases, gradient wrt input to this layer]
        use im2col_bw here to vectorize your computations
        """
        self.dloss = dloss
        # dloss_reshaped = dloss.reshape(self.num_filters, -1)
        dloss_reshaped = dloss.transpose(1, 2, 3, 0).reshape(self.num_filters, -1)
        
        grad_b = np.sum(dloss, axis=(0, 2, 3)).reshape(-1, 1)
        grad_W = np.dot(dloss_reshaped, self.input_cols.T).reshape(self.W.shape)
        
        W_reshaped = self.W.reshape(self.num_filters, -1)
        dX_col = np.dot(W_reshaped.T, dloss_reshaped)
        grad_X = im2col_bw(dX_col, (dloss.shape[0], self.C, self.H, self.Width), self.k_height, self.k_width, self.pad, self.stride)
        
        return grad_W, grad_b, grad_X
    
    def update(self, learning_rate=0.001, momentum_coeff=0.5):
        """
        Update weights and biases with gradients calculated by backward()
        Use the same momentum formula as in Problem 5.
        """
        # Update velocity and then parameters for weights
        self.vW = momentum_coeff * self.vW + learning_rate * self.grad_W
        self.W -= self.vW

        # Update velocity and then parameters for biases
        self.vb = momentum_coeff * self.vb + learning_rate * self.grad_b
        self.b -= self.vb

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
        self.filter_shape = filter_shape
        self.stride = stride
        self.cache = {'max_indices': []} # Cache to store information needed for backward pass

    def forward(self, inputs):
        """
        forward pass of MaxPool
        inputs: (N, C, H, W)
        """
        N, C, H, W = inputs.shape
        FH, FW = self.filter_shape
        stride = self.stride

        # Calculate output dimensions
        H_out = (H - FH) // stride + 1
        W_out = (W - FW) // stride + 1

        # Reshape and stride input to bring non-overlapping regions into separate rows
        # This effectively "tiles" the input so each pooling region is flattened into a row
        strided_input = np.lib.stride_tricks.as_strided(
            inputs,
            shape=(N, C, H_out, W_out, FH, FW),
            strides=(*inputs.strides[:2], inputs.strides[2]*stride, inputs.strides[3]*stride, *inputs.strides[2:]),
            writeable=False
        )

        # Perform max pooling
        pooled_output = np.max(strided_input, axis=(4, 5))

        # Store the indices of max values for the backward pass
        self.cache['input_shape'] = inputs.shape
        max_indices = np.argmax(strided_input.reshape(N, C, H_out, W_out, FH*FW), axis=4)
        self.cache['max_indices'] = np.vstack(np.unravel_index(max_indices, (FH, FW))).T

        return pooled_output
        
        

    def backward(self, dloss):
        """
        dloss is the gradients wrt the output of forward()
        """
        N, C, H_out, W_out = dloss.shape
        FH, FW = self.filter_shape
        stride = self.stride
        input_shape = self.cache['input_shape']
        max_indices = self.cache['max_indices']

        # Initialize gradient array for input with zeros
        grad_input = np.zeros(input_shape)

        # Prepare an array for unraveling max_indices to the shape of grad_input
        unravel_index = np.zeros((max_indices.shape[0], max_indices.shape[1] + 2), dtype=int)
        unravel_index[:, 2:] = max_indices

        # Compute the output indices for each max_index
        n_indices, c_indices = np.meshgrid(np.arange(N), np.arange(C), indexing='ij')
        unravel_index[:, 0] = n_indices.flatten()
        unravel_index[:, 1] = c_indices.flatten()

        # Calculate the correct indices in the flattened input
        flat_indices = np.ravel_multi_index(unravel_index.T, input_shape)

        # Scatter the gradients back to the positions of max values
        # Reshape dloss to match the flat indices shape and then scatter
        dloss_flat = dloss.transpose(0, 1, 2, 3).flatten()
        np.add.at(grad_input.flatten(), flat_indices, dloss_flat)

        return grad_input.reshape(input_shape)

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
        self.inputs = inputs  # Store for use in backward pass
        return np.dot(inputs, self.W) + self.b.T  # Adding bias after dot product
        

    def backward(self, dloss):
        """
        Read Transform.backward()'s docstring in this file
        dloss shape (batch_size, outdim)
        Return [gradient wrt weights, gradient wrt biases, gradient wrt input to this layer]
        """
        # Gradient w.r.t. weights
        grad_W = np.dot(self.inputs.T, dloss)

        # Gradient w.r.t. biases
        grad_b = np.sum(dloss, axis=0, keepdims=True).T

        # Gradient w.r.t. input
        grad_inputs = np.dot(dloss, self.W.T)

        self.grad_W = grad_W
        self.grad_b = grad_b
        return grad_W, grad_b, grad_inputs

    def update(self, learning_rate=0.001, momentum_coeff=0.5):
        """
        Similar to Conv.update()
        """
        if not hasattr(self, 'vW'):
            self.vW = np.zeros_like(self.W)
        if not hasattr(self, 'vb'):
            self.vb = np.zeros_like(self.b)

        # Update velocity and then parameters for weights
        self.vW = momentum_coeff * self.vW + learning_rate * self.grad_W
        self.W -= self.vW

        # Update velocity and then parameters for biases
        self.vb = momentum_coeff * self.vb + learning_rate * self.grad_b
        self.b -= self.vb

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
        # Stable softmax computation
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        self.probabilities = probabilities
        self.labels = labels
        
        # Cross-entropy loss
        loss = -np.sum(labels * np.log(probabilities + 1e-15)) / logits.shape[0]

        if get_predictions:
            return loss, np.argmax(probabilities, axis=1)
        return loss

    def backward(self):
        """
        return shape (batch_size, num_classes)
        Remeber to divide by batch_size so the gradients correspond to the mean loss
        """
        batch_size = self.labels.shape[0]
        # Gradient of softmax-cross entropy loss
        grad = (self.probabilities - self.labels) / batch_size
        return grad

    def getAccu(self):
        """
        Implement as you wish, not autograded.
        """
        correct_preds = np.equal(predictions, np.argmax(labels, axis=1))
        accuracy = np.mean(correct_preds.astype(np.float))
        return accuracy

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
    CIFAR_FILENAME = "/noteboooks/ADL/hw1/cifar10-subset.pkl"
    with open(CIFAR_FILENAME, "rb") as f:
        data = pickle.load(f)

    # preprocess
    trainX = data["trainX"].reshape(-1, 3, 32, 32) / 255.0
    trainy = labels2onehot(data["trainy"])
    testX = data["testX"].reshape(-1, 3, 32, 32) / 255.0
    testy = labels2onehot(data["testy"])
