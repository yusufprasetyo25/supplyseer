import numpy as np

class AdamOptimizer:
    """
    Implements the Adam optimization algorithm.
    """
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

    def update(self, param, grad, m, v):
        self.t += 1
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        param -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon) + 1e-5 * param)
        return param, m, v


class NeuralNet:
    """
    A 2 hidden layer neural network in pure NumPy with both:
    - Gradient Clipping
    - Adam optimizer
    - Built-in training, MSE loss, and tanh & relu activation functions

    Args:
        input_dim: number of input features
        hidden_dim: number of hidden neurons
        output_dim: number of output features (target output)
        activation_function: activation function to use ('tanh' or 'relu')
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, activation_function: str = "tanh"):
        # Initialize weights and biases
        self.initialize_parameters(input_dim, hidden_dim, output_dim)

        # Set activation functions
        self.set_activation_functions(activation_function)

        # Initialize Adam optimizer
        self.optimizer = AdamOptimizer(learning_rate=0.001)
        self.initialize_optimizer_variables()

    def initialize_parameters(self, input_dim: int, hidden_dim: int, output_dim: int):
        """Initialize the weights and biases."""
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2. / hidden_dim)
        self.b2 = np.zeros((1, hidden_dim))

        self.dense_output = np.random.randn(hidden_dim, output_dim) * np.sqrt(2. / hidden_dim)

    def set_activation_functions(self, activation_function: str):
        """Set activation and derivative functions based on the user input."""
        if activation_function == "tanh":
            self.activation_function = self.tanh
            self.derivative_activation = self.tanh_derivative
        elif activation_function == "relu":
            self.activation_function = self.relu
            self.derivative_activation = self.relu_derivative
        else:
            raise ValueError("Unsupported activation function. Choose 'tanh' or 'relu'.")

    def initialize_optimizer_variables(self):
        """Initialize the Adam optimizer variables."""
        self.m_W1, self.v_W1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.m_b1, self.v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.m_W2, self.v_W2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.m_b2, self.v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.m_dense, self.v_dense = np.zeros_like(self.dense_output), np.zeros_like(self.dense_output)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation through the network."""
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.activation_function(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.activation_function(self.z2)

        self.output = np.dot(self.a2, self.dense_output)
        return self.output

    def backprop(self, x: np.ndarray, y: np.ndarray, output: np.ndarray):
        """Backward propagation with gradient clipping and Adam optimizer."""
        n = y.shape[0]

        dz3 = output - y
        d_dense = (1 / n) * np.dot(self.a2.T, dz3)

        da2 = np.dot(dz3, self.dense_output.T)
        dz2 = da2 * self.derivative_activation(self.z2)

        dW2 = (1 / n) * np.dot(self.a1.T, dz2)
        db2 = (1 / n) * np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.derivative_activation(self.z1)

        dW1 = (1 / n) * np.dot(x.T, dz1)
        db1 = (1 / n) * np.sum(dz1, axis=0, keepdims=True)

        # Gradient clipping
        clip_value = 2
        dW1 = np.clip(dW1, -clip_value, clip_value)
        db1 = np.clip(db1, -clip_value, clip_value)
        dW2 = np.clip(dW2, -clip_value, clip_value)
        db2 = np.clip(db2, -clip_value, clip_value)
        d_dense = np.clip(d_dense, -clip_value, clip_value)

        # Update parameters using Adam optimizer
        self.W1, self.m_W1, self.v_W1 = self.optimizer.update(self.W1, dW1, self.m_W1, self.v_W1)
        self.b1, self.m_b1, self.v_b1 = self.optimizer.update(self.b1, db1, self.m_b1, self.v_b1)
        self.W2, self.m_W2, self.v_W2 = self.optimizer.update(self.W2, dW2, self.m_W2, self.v_W2)
        self.b2, self.m_b2, self.v_b2 = self.optimizer.update(self.b2, db2, self.m_b2, self.v_b2)
        self.dense_output, self.m_dense, self.v_dense = self.optimizer.update(self.dense_output, d_dense, self.m_dense, self.v_dense)

    def compute_mse(self, true: np.ndarray, pred: np.ndarray) -> float:
        """Compute Mean Squared Error loss."""
        return np.mean((true - pred) ** 2)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, learning_rate: float = 0.0001, batch_size: int = 32):
        """
        Train the neural network on the provided dataset.

        Args:
            X: Training data.
            y: Target values.
            epochs: Number of training epochs.
            learning_rate: Learning rate for optimization.
            batch_size: Size of each training batch.
        """
        self.optimizer.learning_rate = learning_rate
        n_samples = X.shape[0]

        for epoch in range(epochs):
            # Shuffle data at the start of each epoch
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                output = self.forward(X_batch)
                self.backprop(X_batch, y_batch, output)

            # Print loss every 10 epochs
            if epoch % 10 == 0:
                loss = self.compute_mse(y, self.forward(X))
                print(f"Epoch {epoch}, Loss: {loss}")

        return self.forward(X)

class MixDensityNet:
    """
    Univariate 2 hidden layer Mixture Density Neural Network in pure NumPy.
    This MDN is capable of modeling a univariate target variable using multiple Gaussian mixtures.

    Args:
        input_dim: number of input features
        hidden_dim: number of hidden neurons
        n_mixtures: number of output mixtures (target output)
        activation_function: activation function to use ('tanh', 'relu', 'leaky_relu', 'sigmoid')
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_mixtures: int, activation_function: str = "tanh"):
        # Initialize network parameters
        self.initialize_parameters(input_dim, hidden_dim, n_mixtures)
        # Set activation functions
        self.set_activation_functions(activation_function)
        # Initialize Adam optimizer
        self.optimizer = AdamOptimizer(learning_rate=0.001)
        self.initialize_optimizer_variables()

    def initialize_parameters(self, input_dim: int, hidden_dim: int, n_mixtures: int):
        """Initialize the weights and biases using Xavier (Glorot) initialization."""
        limit_W1 = np.sqrt(6 / (input_dim + hidden_dim))
        self.W1 = np.random.uniform(-limit_W1, limit_W1, (input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))

        limit_W2 = np.sqrt(6 / (hidden_dim + hidden_dim))
        self.W2 = np.random.uniform(-limit_W2, limit_W2, (hidden_dim, hidden_dim))
        self.b2 = np.zeros((1, hidden_dim))

        limit_W_pi = np.sqrt(6 / (hidden_dim + n_mixtures))
        self.W_pi = np.random.uniform(-limit_W_pi, limit_W_pi, (hidden_dim, n_mixtures))
        self.b_pi = np.zeros((1, n_mixtures))

        limit_W_mu = np.sqrt(6 / (hidden_dim + n_mixtures))
        self.W_mu = np.random.uniform(-limit_W_mu, limit_W_mu, (hidden_dim, n_mixtures))
        self.b_mu = np.zeros((1, n_mixtures))

        limit_W_sigma = np.sqrt(6 / (hidden_dim + n_mixtures))
        self.W_sigma = np.random.uniform(-limit_W_sigma, limit_W_sigma, (hidden_dim, n_mixtures))
        self.b_sigma = np.zeros((1, n_mixtures))

    def set_activation_functions(self, activation_function: str):
        """Set activation and derivative functions based on user input."""
        activation_functions = {
            "tanh": (self.tanh, self.tanh_derivative),
            "relu": (self.relu, self.relu_derivative),
            "leaky_relu": (self.leaky_relu, self.leaky_relu_derivative),
            "sigmoid": (self.sigmoid, self.sigmoid_derivative)
        }
        if activation_function in activation_functions:
            self.activation_function, self.derivative_activation = activation_functions[activation_function]
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")

    def initialize_optimizer_variables(self):
        """Initialize the Adam optimizer variables."""
        self.m_W1, self.v_W1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.m_b1, self.v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.m_W2, self.v_W2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.m_b2, self.v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.m_W_pi, self.v_W_pi = np.zeros_like(self.W_pi), np.zeros_like(self.W_pi)
        self.m_b_pi, self.v_b_pi = np.zeros_like(self.b_pi), np.zeros_like(self.b_pi)
        self.m_W_mu, self.v_W_mu = np.zeros_like(self.W_mu), np.zeros_like(self.W_mu)
        self.m_b_mu, self.v_b_mu = np.zeros_like(self.b_mu), np.zeros_like(self.b_mu)
        self.m_W_sigma, self.v_W_sigma = np.zeros_like(self.W_sigma), np.zeros_like(self.W_sigma)
        self.m_b_sigma, self.v_b_sigma = np.zeros_like(self.b_sigma), np.zeros_like(self.b_sigma)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def leaky_relu(x):
        return np.maximum(0.1 * x, x)

    @staticmethod
    def leaky_relu_derivative(x):
        return np.where(x > 0, 1, 0.1)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 - sig)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def softplus(x):
        return np.log(1 + np.exp(x))


    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward propagation through the network."""
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.activation_function(self.z1)

        # Batch norm
        self.a1 = self.batch_norm(self.a1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.activation_function(self.z2)

        # Batch norm
        self.a2 = self.batch_norm(self.a2)

        pi = self.softmax(np.dot(self.a2, self.W_pi) + self.b_pi)
        mu = np.dot(self.a2, self.W_mu) + self.b_mu
        # sigma = np.exp(np.dot(self.a2, self.W_sigma) + self.b_sigma)
        sigma = self.softplus(np.dot(self.a2, self.W_sigma) + self.b_sigma) + 1e-6  # Add small epsilon
        sigma = np.clip(sigma, 1e-2, 1e2)

        return pi, mu, sigma

    def compute_loss(self, y, pi, mu, sigma):
        """Compute the negative log-likelihood loss for the MDN."""
        y = y.reshape(-1, 1)  # Ensure y has the correct shape

        # Compute normal PDF values for each component and data point
        coeff = 1 / (np.sqrt(2 * np.pi) * sigma)
        exponent = -0.5 * ((y - mu) / sigma) ** 2
        pdf = coeff * np.exp(exponent)  # Shape: (n_samples, n_components)

        # Weight by mixing coefficients and sum over components
        weighted_pdf = pdf * pi
        total_pdf = np.sum(weighted_pdf, axis=1)  # Shape: (n_samples,)

        # Compute negative log-likelihood
        nll_loss = -np.mean(np.log(total_pdf + 1e-8))  # Add epsilon to prevent log(0)

        return nll_loss

        
    def batch_norm(self, x, epsilon=1e-5):
        mean = np.mean(x, axis=0, keepdims=True)
        variance = np.var(x, axis=0, keepdims=True)
        return (x - mean) / np.sqrt(variance + epsilon)

    def backprop(self, x: np.ndarray, y: np.ndarray, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
        """Backward propagation with vectorized gradient computation and Adam optimizer."""
        n = y.shape[0]
        y = y.reshape(-1, 1)  # Ensure y has the correct shape

        # Compute responsibilities (gamma)
        normals = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((y - mu) / sigma) ** 2)
        P_y_given_x = np.sum(pi * normals, axis=1, keepdims=True) + 1e-8  # Avoid division by zero
        gamma = (pi * normals) / P_y_given_x  # Shape: (n_samples, n_components)

        # Compute gradients
        pi_grad = gamma - pi  # Shape: (n_samples, n_components)
        mu_grad = gamma * (mu - y) / (sigma ** 2)
        sigma_grad = gamma * (((mu - y) ** 2) / (sigma ** 3) - 1 / sigma)

        # Clip gradients if necessary
        clip_value = 1.0  # Adjust as needed
        pi_grad = np.clip(pi_grad, -clip_value, clip_value)
        mu_grad = np.clip(mu_grad, -clip_value, clip_value)
        sigma_grad = np.clip(sigma_grad, -clip_value, clip_value)

        # Gradients w.r.t. the outputs of the last hidden layer
        da2 = (np.dot(pi_grad, self.W_pi.T) +
            np.dot(mu_grad, self.W_mu.T) +
            np.dot(sigma_grad, self.W_sigma.T))  # Shape: (n_samples, hidden_dim)
        dz2 = da2 * self.derivative_activation(self.z2)

        # Gradients for W_pi, W_mu, W_sigma
        dW_pi = np.dot(self.a2.T, pi_grad)
        db_pi = np.sum(pi_grad, axis=0, keepdims=True)
        dW_mu = np.dot(self.a2.T, mu_grad)
        db_mu = np.sum(mu_grad, axis=0, keepdims=True)
        dW_sigma = np.dot(self.a2.T, sigma_grad)
        db_sigma = np.sum(sigma_grad, axis=0, keepdims=True)

        # Update output layer weights and biases
        self.W_pi, self.m_W_pi, self.v_W_pi = self.optimizer.update(self.W_pi, dW_pi, self.m_W_pi, self.v_W_pi)
        self.b_pi, self.m_b_pi, self.v_b_pi = self.optimizer.update(self.b_pi, db_pi, self.m_b_pi, self.v_b_pi)
        self.W_mu, self.m_W_mu, self.v_W_mu = self.optimizer.update(self.W_mu, dW_mu, self.m_W_mu, self.v_W_mu)
        self.b_mu, self.m_b_mu, self.v_b_mu = self.optimizer.update(self.b_mu, db_mu, self.m_b_mu, self.v_b_mu)
        self.W_sigma, self.m_W_sigma, self.v_W_sigma = self.optimizer.update(self.W_sigma, dW_sigma, self.m_W_sigma, self.v_W_sigma)
        self.b_sigma, self.m_b_sigma, self.v_b_sigma = self.optimizer.update(self.b_sigma, db_sigma, self.m_b_sigma, self.v_b_sigma)

        # Backpropagate to hidden layers
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.derivative_activation(self.z1)

        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        dW1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Clip gradients for hidden layers
        dW2 = np.clip(dW2, -clip_value, clip_value)
        db2 = np.clip(db2, -clip_value, clip_value)
        dW1 = np.clip(dW1, -clip_value, clip_value)
        db1 = np.clip(db1, -clip_value, clip_value)

        # Update hidden layer weights and biases
        self.W2, self.m_W2, self.v_W2 = self.optimizer.update(self.W2, dW2, self.m_W2, self.v_W2)
        self.b2, self.m_b2, self.v_b2 = self.optimizer.update(self.b2, db2, self.m_b2, self.v_b2)
        self.W1, self.m_W1, self.v_W1 = self.optimizer.update(self.W1, dW1, self.m_W1, self.v_W1)
        self.b1, self.m_b1, self.v_b1 = self.optimizer.update(self.b1, db1, self.m_b1, self.v_b1)


    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int = 1000, learning_rate: float = 0.001):
        """Train the Mixture Density Network on the provided dataset."""
        self.optimizer.learning_rate = learning_rate
        for epoch in range(epochs):
            # Forward pass
            pi, mu, sigma = self.forward(x_train)
            # Compute loss
            loss = self.compute_loss(y_train, pi, mu, sigma)

            # Print loss every 10 epochs
            if epoch % 10 == 0:
                param_norm = np.linalg.norm(self.W1)
                # print(f"Epoch {epoch}, Loss: {loss}, W1 Norm: {param_norm}")
                print(f"Epoch {epoch}, Loss: {loss}")
                print(f"W1 Norm: {np.linalg.norm(self.W1)}, W2 Norm: {np.linalg.norm(self.W2)}")
                print(f"Sigma Min: {sigma.min()}, Sigma Max: {sigma.max()}, Sigma Mean: {sigma.mean()}")
                print(f"Mu Min: {mu.min()}, Mu Max: {mu.max()}, Mu Mean: {mu.mean()}")
                # print(f"Epoch {epoch}, Loss: {loss}")

            # Backpropagation pass
            self.backprop(x_train, y_train, pi, mu, sigma)
