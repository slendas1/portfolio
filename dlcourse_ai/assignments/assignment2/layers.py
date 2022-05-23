import numpy as np
from copy import deepcopy
from math import sqrt

def D_array_for_2D(array):
    if(array.ndim == 1):
        return array[np.newaxis,:].copy()
    else:
        return array.copy()

def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength * np.sum(np.square(W))
    grad = W * (2 * reg_strength)

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    """
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
    """
    # TODO: Copy from the previous assignment
    assert isinstance(predictions, np.ndarray)

    is_1D = (predictions.ndim == 1)
    predictions = D_array_for_2D(predictions)
    batch_size = predictions.shape[0]

    predictions -= np.max(predictions, axis=1)[:, np.newaxis]
    predictions = np.exp(predictions)

    probs = predictions / np.sum(predictions, axis=1)[:, np.newaxis]

    dprediction = probs.copy()
    dprediction[(np.arange(batch_size), target_index)] -= 1
    dprediction /= batch_size
    if (is_1D):
        dprediction = np.squeeze(dprediction, axis=0)

    loss_arr = -np.log(probs)
    loss = np.sum(loss_arr[(np.arange(batch_size), target_index)]) / batch_size
    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.grad_fn = None
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        grad_fn1 = np.zeros_like(X)
        grad_fn1[X > 0] = 1

        self.grad_fn = grad_fn1

        answer = X.copy()
        answer[X < 0] = 0
        return answer

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = d_out * self.grad_fn
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(np.random.normal(0, sqrt(2 / n_input), (n_input, n_output)))
        self.B = Param(np.random.normal(0, sqrt(2 / n_input), n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = deepcopy(X)

        return np.matmul(X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        d_input = np.matmul(d_out, self.W.value.T)
        self.W.grad += np.matmul(self.X.T, d_out)
        self.B.grad += np.sum(d_out, axis=0)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
