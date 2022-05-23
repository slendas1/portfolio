import numpy as np
from utils import transfrom_1Darray_to_2D
from copy import deepcopy


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W
    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''

    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    assert isinstance(predictions, np.ndarray)

    initial_num_of_dimensions = predictions.ndim
    x = transfrom_1Darray_to_2D(predictions)

    max_ = np.max(x, axis=1)
    x = x - max_[:, np.newaxis]
    x = np.exp(x)
    sum_ = np.sum(x, axis=1)
    x = x / sum_[:, np.newaxis]
    if initial_num_of_dimensions == 1:
        x = np.squeeze(x, axis=0)
    return x


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    assert isinstance(probs, np.ndarray)
    assert isinstance(target_index, np.ndarray)

    X = transfrom_1Darray_to_2D(probs)
    y = target_index
    batch_size, number_of_features = X.shape



    loss = X[(np.arange(batch_size), y)]
    loss = -np.log(loss)
    loss = np.sum(loss) / batch_size

    return loss


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO copy from the previous assignment
    assert predictions.ndim <= 2, "predictions must be a numpy array with no more than two dimensions"
    assert isinstance(predictions, np.ndarray)
    initial_num_of_dimensions = predictions.ndim

    probs = softmax(predictions)
    probs = transfrom_1Darray_to_2D(probs)
    loss = cross_entropy_loss(probs, target_index)

    predictions = transfrom_1Darray_to_2D(predictions)
    batch, N = predictions.shape

    dprediction1 = np.zeros((batch, N))
    dprediction1[(np.arange(batch), target_index)] = probs[(np.arange(batch), target_index)].copy()
    dprediction2 = probs.copy()
    dprediction2 *= probs[(np.arange(batch), target_index)][:, np.newaxis]
    dprediction = dprediction1 - dprediction2
    dprediction = -dprediction / probs[(np.arange(batch), target_index)][:, np.newaxis]
    dprediction /= batch

    if initial_num_of_dimensions == 1:
        dprediction = np.squeeze(dprediction, axis=0)

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self._value = value
        self._grad = np.zeros_like(value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, var):
        self._value = var

    @value.deleter
    def value(self):
        del self._value

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, var):
        assert self.grad.shape == var.shape, "Shape of gradient must not be changed. However, shape of " \
                                             f"gradient={self.grad.shape}, shape of the value you tried to" \
                                             f" assign={var.shape}"

        self._grad = var

    @grad.deleter
    def grad(self):
        del self._grad

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        self.zero_grad = X >= 0
        return np.maximum(X, np.zeros_like(X))

    def backward(self, d_out):
        # TODO copy from the previous assignment
        d_result = d_out * self.zero_grad
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = deepcopy(X)
        return np.matmul(X, self.W.value) + self.B.value

    def backward(self, d_out):
        # TODO copy from the previous assignment

        batch_size, n_output = d_out.shape
        _, n_input = self.X.shape

        self.B.grad += np.sum(d_out, axis=0)

        d_input = np.zeros((batch_size, n_input))
        for i in range(batch_size):
            self.W.grad += np.repeat(self.X[i, :][:, np.newaxis], n_output, axis=1) * d_out[[i], :]
            d_input[i, :] += np.sum(self.W.value * d_out[[i], :], axis=1)
        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding

    def forward(self, X):
        self.X = X
        batch_size, height, width, channels = X.shape

        out_height = height - self.filter_size + 1 + 2 * self.padding
        out_width = width - self.filter_size + 1 + 2 * self.padding
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        self.X_padded = \
            np.pad(X, [(0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)], constant_values=[(0, 0), (0, 0), (0, 0), (0, 0)])
        X_out = np.zeros((batch_size, out_height, out_width, self.out_channels))
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                y_left, y_right = y, y + self.filter_size
                x_left, x_right = x, x + self.filter_size
                y_cor, x_cor = np.meshgrid(np.arange(y_left, y_right), np.arange(x_left, x_right))
                # y_cor, x_cor = np.ix_(np.arange(y_left, y_right), np.arange(x_left, x_right))
                input = self.X_padded[:, y_cor, x_cor, :].reshape(batch_size, -1)
                X_out[:, y, x, :] = np.matmul(input, self.W.value.reshape(-1, self.out_channels))
        X_out += self.B.value
        return X_out

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        d_input = np.zeros((batch_size, height + self.padding * 2, width + self.padding * 2, channels))
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                y_left, y_right = y, y + self.filter_size
                x_left, x_right = x, x + self.filter_size
                y_cor, x_cor = np.meshgrid(np.arange(y_left, y_right), np.arange(x_left, x_right))
                y_cor, x_cor = y_cor.flatten(), x_cor.flatten()
                A = self.W.value.reshape(-1, out_channels)
                B = np.moveaxis(d_out[:, y, x, :], 0, 1)
                result = np.matmul(A, B).reshape((self.filter_size, self.filter_size, self.in_channels, batch_size))
                d_input[:, y_cor, x_cor, :] += np.moveaxis(result, -1, 0).reshape((batch_size, -1, self.in_channels))
                # that's really funny because array indexing creates a copy, it's not a view, but somehow in this
                # particular case with += it works as view

                input = self.X_padded[:, y_cor, x_cor, :].reshape(batch_size, -1)
                A = np.moveaxis(input, 0, -1)
                B = d_out[:, y, x, :]
                self.W.grad += np.matmul(A, B).reshape((self.filter_size, self.filter_size, self.in_channels, self.out_channels))

                self.B.grad = np.sum(d_out, (0, 1, 2))
        y = np.arange(self.padding, height + self.padding)
        x = np.arange(self.padding, width + self.padding)
        y, x = np.ix_(y, x)
        return d_input[:, y, x, :]

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        self.X = X
        assert height % self.stride == 0 and width % self.stride == 0
        out_height, out_width = int(height / self.stride), int(width / self.stride)
        result = np.zeros((batch_size, out_height, out_width, channels))
        for y in range(out_height):
            for x in range(out_width):
                y_left, y_right = y * self.stride, (y + 1) * self.stride
                x_left, x_right = x * self.stride, (x + 1) * self.stride
                y_cor, x_cor = np.ix_(np.arange(y_left, y_right), np.arange(x_left, x_right))
                result[:, y, x, :] = np.amax(X[:, y_cor, x_cor, :], (1, 2))
        return result

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        d_in = np.zeros((batch_size, height, width, channels))
        out_height, out_width = int(height / self.stride), int(width / self.stride)
        for y in range(out_height):
            for x in range(out_width):
                for batch_ind in range(batch_size):
                    for channel_ind in range(channels):
                        y_left, y_right = y * self.stride, (y + 1) * self.stride
                        x_left, x_right = x * self.stride, (x + 1) * self.stride
                        y_cor, x_cor = np.ix_(np.arange(y_left, y_right), np.arange(x_left, x_right))
                        a = np.argmax(self.X[batch_ind, y_cor, x_cor, channel_ind].reshape(-1), axis=0)
                        y_max, x_max = np.unravel_index(a, shape=(self.stride, self.stride))
                        d_in[batch_ind, y_max, x_max, channel_ind] += d_out[batch_ind, y, x, channel_ind]
        return d_in

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        self.X_shape = (batch_size, height, width, channels)
        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        return X.reshape(batch_size, -1)

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
