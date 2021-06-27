import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from utils import sum_gt_zero, xor


num_cases = 3  # number of auto-generated cases
num_epochs = 1000  # number of epochs for training
classification_function = sum_gt_zero  # selects sum_gt_zero as the classification function

# Setting the random seed of numpy's random library for reproducibility reasons
np.random.seed(0)

# Creating the dataset
inputs = 5.0 * (-1.0 + 2.0 * np.random.rand(num_cases, 2))
expected_outputs = np.array([classification_function(x) for x in inputs])

# Separating the dataset into positive and negative samples
positives_indices = np.where(expected_outputs >= 0.5)
negatives_indices = np.where(expected_outputs < 0.5)
positives = inputs[positives_indices]
negatives = inputs[negatives_indices]

# Creating and training the neural network
neural_network = NeuralNetwork(2, 10, 1, 6.0)
costs = np.zeros(num_epochs)
inputs_nn = inputs.T

print("inputs_nn", inputs_nn)

neural_network.back_propagation(inputs_nn, expected_outputs)
