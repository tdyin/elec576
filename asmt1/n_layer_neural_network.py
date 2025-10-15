import numpy as np
from sklearn import datasets
from three_layer_neural_network import NeuralNetwork, generate_data, plot_decision_boundary


def generate_iris_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y


class Layer:
    """
    Represents a single layer in the neural network.
    Handles feedforward and backpropagation for one layer.
    """

    def __init__(self, input_dim, output_dim, actFun_type='tanh', is_output_layer=False):
        """
        Initialize a layer with weights and biases.
        :param input_dim: dimension of input to this layer
        :param output_dim: dimension of output from this layer
        :param actFun_type: type of activation function ('tanh', 'sigmoid', 'relu')
        :param is_output_layer: whether this is the output layer (uses softmax instead of activation)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.actFun_type = actFun_type
        self.is_output_layer = is_output_layer

        # Initialize weights and biases
        self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.b = np.zeros((1, output_dim))

        # Cache for storing intermediate values during forward pass
        self.z = None  # pre-activation
        self.a = None  # post-activation
        self.input = None  # input to this layer

    def actFun(self, z):
        """Compute activation function"""
        if self.actFun_type == 'tanh':
            return np.tanh(z)
        elif self.actFun_type == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.actFun_type == 'relu':
            return np.maximum(0, z)
        else:
            raise ValueError(f"Unknown activation function: {self.actFun_type}")

    def diff_actFun(self, z):
        """Compute derivative of activation function"""
        if self.actFun_type == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif self.actFun_type == 'sigmoid':
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig)
        elif self.actFun_type == 'relu':
            dz = np.array(z, copy=True)
            dz[z <= 0] = 0
            dz[z > 0] = 1
            return dz
        else:
            raise ValueError(f"Unknown activation function: {self.actFun_type}")

    def feedforward(self, X):
        """
        Perform feedforward pass through this layer.
        :param X: input to this layer (batch_size x input_dim)
        :return: output from this layer (batch_size x output_dim)
        """
        self.input = X
        self.z = X.dot(self.W) + self.b

        if self.is_output_layer:
            # Output layer uses softmax
            exp_scores = np.exp(self.z)
            self.a = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        else:
            # Hidden layers use activation function
            self.a = self.actFun(self.z)

        return self.a

    def backprop(self, delta_next, reg_lambda=0.01):
        """
        Perform backpropagation through this layer.
        :param delta_next: gradient from the next layer (batch_size x output_dim)
        :param reg_lambda: regularization coefficient
        :return: delta for previous layer (batch_size x input_dim)
        """
        num_examples = self.input.shape[0]

        if not self.is_output_layer:
            # For hidden layers, multiply by derivative of activation function
            delta = delta_next * self.diff_actFun(self.z)
        else:
            # For output layer, delta_next is already delta
            delta = delta_next

        # Calculate gradients
        self.dW = (self.input.T).dot(delta)
        self.db = np.sum(delta, axis=0, keepdims=True)

        # Add regularization to weight gradients
        self.dW += reg_lambda * self.W

        # Calculate delta for previous layer
        delta_prev = delta.dot(self.W.T)

        return delta_prev

    def update_weights(self, epsilon):
        """
        Update weights and biases using gradients.
        :param epsilon: learning rate
        """
        self.W += -epsilon * self.dW
        self.b += -epsilon * self.db


class DeepNeuralNetwork(NeuralNetwork):
    def __init__(
            self,
            nn_input_dim,
            nn_output_dim,
            num_hidden_layers=1,
            hidden_dim=3,
            actFun_type='tanh',
            reg_lambda=0.01,
            seed=0,
    ):
        # Store meta
        self.nn_input_dim = nn_input_dim
        self.nn_output_dim = nn_output_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        self.seed = seed

        # Initialize random seed
        np.random.seed(seed)

        # Create layers
        self.layers = []

        # First hidden layer (input -> hidden)
        self.layers.append(Layer(nn_input_dim, hidden_dim, actFun_type, is_output_layer=False))

        # Additional hidden layers (hidden -> hidden)
        for i in range(num_hidden_layers - 1):
            self.layers.append(Layer(hidden_dim, hidden_dim, actFun_type, is_output_layer=False))

        # Output layer (hidden -> output)
        self.layers.append(Layer(hidden_dim, nn_output_dim, actFun_type, is_output_layer=True))

    def feedforward(self, X):
        """
        Feedforward through all layers using Layer.feedforward
        :param X: input data
        :return: None (stores probabilities in self.probs)
        """
        a = X
        for layer in self.layers:
            a = layer.feedforward(a)

        # Store final output as probabilities
        self.probs = a
        return None

    def calculate_loss(self, X, y):
        """
        Calculate loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        """
        num_examples = len(X)
        self.feedforward(X)

        # Calculate cross-entropy loss
        correct_logprobs = -np.log(self.probs[range(num_examples), y])
        data_loss = np.sum(correct_logprobs)

        # Add regularization term to loss
        reg_loss = 0
        for layer in self.layers:
            reg_loss += np.sum(np.square(layer.W))
        data_loss += self.reg_lambda / 2 * reg_loss

        return (1. / num_examples) * data_loss

    def predict(self, X):
        """
        Predict labels for input data
        :param X: input data
        :return: predicted labels
        """
        self.feedforward(X)
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        """
        Backpropagation through all layers using Layer.backprop
        :param X: input data
        :param y: given labels
        :return: None (gradients stored in each layer)
        """
        num_examples = len(X)

        # Initial delta for output layer (derivative of cross-entropy loss with softmax)
        delta = self.probs.copy()
        delta[range(num_examples), y] -= 1
        delta /= num_examples

        # Backpropagate through layers in reverse order
        for layer in reversed(self.layers):
            delta = layer.backprop(delta, self.reg_lambda)

        return None

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        """
        Train the network using gradient descent
        :param X: input data
        :param y: given labels
        :param epsilon: learning rate
        :param num_passes: number of iterations
        :param print_loss: whether to print loss during training
        :return: None
        """
        for i in range(num_passes):
            # Forward propagation
            self.feedforward(X)

            # Backpropagation
            self.backprop(X, y)

            # Update weights for all layers
            for layer in self.layers:
                layer.update_weights(epsilon)

            # Print loss
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

        return None


def main():
    # Generate Make-Moons dataset
    # X, y = generate_data()
    # Or generate other dataset
    X, y = generate_iris_data()

    # Train a deep neural network
    # for hidden_layers in [1, 3, 5, 10]:
    #     print(f"\nTraining a network with { hidden_layers } hidden layers:")
    #     model = DeepNeuralNetwork(nn_input_dim=2, nn_output_dim=2,
    #                               num_hidden_layers=hidden_layers, hidden_dim=5,
    #                               actFun_type='tanh', reg_lambda=0.01, seed=0)
    #     model.fit_model(X, y)
    #     model.visualize_decision_boundary(X, y)
    #
    # for hidden_units in [10, 20, 50, 100]:
    #     print(f"\nTraining a network with { hidden_units } hidden units:")
    #     model = DeepNeuralNetwork(nn_input_dim=2, nn_output_dim=2,
    #                               num_hidden_layers=3, hidden_dim=hidden_units,
    #                               actFun_type='tanh', reg_lambda=0.01, seed=0)
    #     model.fit_model(X, y)
    #     model.visualize_decision_boundary(X, y)
    #
    # for actFun_type in ['tanh', 'sigmoid', 'relu']:
    #     print(f"\nTraining a network with activation function: { actFun_type }")
    #     model = DeepNeuralNetwork(nn_input_dim=2, nn_output_dim=2,
    #                               num_hidden_layers=3, hidden_dim=5,
    #                               actFun_type=actFun_type, reg_lambda=0.01, seed=0)
    #     model.fit_model(X, y)
    #     model.visualize_decision_boundary(X, y)

    for [layers, units, actFun_type] in [
        [3, 10, 'tanh'],
        [3, 10, 'sigmoid'],
        [3, 10, 'relu'],
        [5, 20, 'tanh'],
        [5, 20, 'sigmoid'],
        [5, 20, 'relu'],
        [10, 50, 'tanh'],
        [10, 50, 'sigmoid'],
        [10, 50, 'relu'],
    ]:
        print(
            f"\nTraining a network with {layers} hidden layers, {units} hidden units and activation function: {actFun_type}")
        model = DeepNeuralNetwork(nn_input_dim=4, nn_output_dim=3,
                                  num_hidden_layers=layers, hidden_dim=units,
                                  actFun_type=actFun_type, reg_lambda=0.01, seed=0)
        model.fit_model(X, y)


if __name__ == "__main__":
    main()
