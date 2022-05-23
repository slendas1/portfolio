import numpy as np
from copy import deepcopy

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        input_width, input_height, input_channel = input_shape
        self.layers = ['Conv1', 'Relu1', 'MaxPool1', 'Conv2', 'Relu2', 'MaxPool2', 'Flatten', 'FC']
        self.Conv1 = ConvolutionalLayer(input_channel, conv1_channels, 3, 1)
        self.Relu1 = ReLULayer()
        self.MaxPool1 = MaxPoolingLayer(4, 4)
        self.Conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1)
        self.Relu2 = ReLULayer()
        self.MaxPool2 = MaxPoolingLayer(4, 4)
        self.Flatten = Flattener()
        self.FC = FullyConnectedLayer((input_width * input_height * conv2_channels) >> 8, n_output_classes)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        for name_of_layer in self.layers:
            layer = getattr(self, name_of_layer)
            for param in layer.params().values():
                param.grad = np.zeros_like(param.value)

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        input_value = X.copy()
        for name_of_layer in self.layers:
            layer = getattr(self, name_of_layer)
            input_value = deepcopy(layer.forward(input_value))
        loss_cross_entropy, dpred = softmax_with_cross_entropy(input_value, y)

        # backward pass for cross entropy
        input_grad = deepcopy(dpred)
        for name_of_layer in self.layers[::-1]:
            layer = getattr(self, name_of_layer)
            input_grad = layer.backward(input_grad)

        return loss_cross_entropy

    def predict(self, X):
        # You can probably copy the code from previous assignment
        input_value = X
        for name_of_layer in self.layers:
            layer = getattr(self, name_of_layer)
            input_value = layer.forward(input_value)

        pred = np.argmax(input_value, axis=1)
        return pred

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        for name_of_layer in self.layers:
            layer = getattr(self, name_of_layer)
            for name, value in layer.params().items():
                name = f"{name_of_layer}_{name}"
                result[name] = value
        return result
