import numpy as np

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

    Functions:
        forward: does the forward propagation
        backprop: does the backpropagation with Adam optimizer
        train: training on X and y sets provided by user

    """
    def __init__(self, input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 activation_function: str ="tanh"):
        
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.b2 = np.zeros((1, hidden_dim))

        self.dense_output = np.random.randn(hidden_dim, output_dim) *  np.sqrt(2. / input_dim)
        self.activation_function = self.tanh if activation_function == "tanh" else self.relu
        self.derivative_activation = self.tanh_derivative if activation_function == "tanh" else self.relu_derivative
        
        self.m_W1, self.v_W1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.m_b1, self.v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.m_W2, self.v_W2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.m_b2, self.v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.m_dense, self.v_dense = np.zeros_like(self.dense_output), np.zeros_like(self.dense_output)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0 
    
    def _adam_update(self, param, grad, m, v):
        self.t += 1
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * (grad**2)
        m_hat = m / (1 - self.beta1**self.t)
        v_hat = v / (1 - self.beta2**self.t)
        update = m_hat / (np.sqrt(v_hat) + self.epsilon)

        return param - self.learning_rate * update, m, v

    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def forward(self, x):

        self.z1 = np.add(self.b1, np.dot(x, self.W1))
        self.a1 = self.activation_function(self.z1)

        self.z2 = np.add(self.b2, np.dot(self.a1, self.W2))
        self.a2 = self.activation_function(self.z2)

        self.output = np.dot(self.a2, self.dense_output)

        return self.output
    
    def backprop(self, x, y, output, learning_rate=0.001):

        n = y.shape[0]

        dz3 = np.subtract(output, y)
        d_dense = (1 / n) * np.dot(self.a2.T, dz3)

        da2 = np.dot(dz3, self.dense_output.T)
        dz2 = da2 * self.derivative_activation(self.z2)

        dW2 = (1 / n) * np.dot(self.a1.T, dz2)
        db2 = (1 / n) * np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.derivative_activation(self.z1)

        dW1 = (1 / n) * np.dot(x.T, dz1)
        db1 = (1 / n) * np.sum(dz1, axis=0, keepdims=True)

        clip_value = 2
        dW1 = np.clip(dW1, -clip_value, clip_value)
        db1 = np.clip(db1, -clip_value, clip_value)
        dW2 = np.clip(dW2, -clip_value, clip_value)
        db2 = np.clip(db2, -clip_value, clip_value)
        d_dense = np.clip(d_dense, -clip_value, clip_value)

        self.W1, self.m_W1, self.v_W1 = self._adam_update(self.W1, dW1, self.m_W1, self.v_W1)
        self.b1, self.m_b1, self.v_b1 = self._adam_update(self.b1, db1, self.m_b1, self.v_b1)
        self.W2, self.m_W2, self.v_W2 = self._adam_update(self.W2, dW2, self.m_W2, self.v_W2)
        self.b2, self.m_b2, self.v_b2 = self._adam_update(self.b2, db2, self.m_b2, self.v_b2)
        self.dense_output, self.m_dense, self.v_dense = self._adam_update(self.dense_output, d_dense, self.m_dense, self.v_dense)
    

    def compute_mse(self, true, pred):
        return np.mean(((true - pred)**2))

    def train(self, X, y, epochs=100, learning_rate=0.0001):
        self.learning_rate = learning_rate
        for epoch in range(epochs):
            output = self.forward(X)
            self.backprop(X, y, learning_rate)
            
            if epoch % 10 == 0:
                loss = self.compute_mse(y, output)
                print(f"Epoch {epoch}, Loss: {loss}")
        return output




class MixDensityNet:
    """
    Univariate 2 hidden layer Mixture Density Neural Network in pure NumPy with simple 
    settings to have a working MDN model. Since this is a univariate model it will only
    handle 1 target feature but as many input features as you want.

    Args:
        input_dim: number of input features
        hidden_dim: number of hidden neurons
        n_mixtures: number of output mixtures (target output)

    Functions:
        forward: does the forward propagation
        backprop: does the backpropagation with Adam optimizer
        train: training on X and y sets provided by user
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_mixtures: int, activation_function: str):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b2 = np.zeros((1, hidden_dim))

        self.W_pi = np.random.randn(hidden_dim, n_mixtures) * 0.01
        self.b_pi = np.zeros((1, n_mixtures))

        self.W_mu = np.random.randn(hidden_dim, n_mixtures) * 0.01
        self.b_mu = np.zeros((1, n_mixtures))

        self.W_sigma = np.random.randn(hidden_dim, n_mixtures) * 0.01
        self.b_sigma = np.zeros((1, n_mixtures))

        self.activation_function = self.grab_activation(activation_function)
        self.derivative_activation = self.grab_derivative_activation(activation_function)

    def grab_activation(self, input_name: str):
        match input_name:
            case "tanh":
                return self.tanh
            case "relu":
                return self.relu
            case "leaky_relu":
                return self.leaky_relu
            case "sigmoid":
                return self.sigmoid
    
    def grab_derivative_activation(self, input_name: str):
        match input_name:
            case "tanh":
                return self.tanh_derivative
            case "relu":
                return self.relu_derivative
            case "leaky_relu":
                return self.leaky_relu_derivative
            case "sigmoid":
                return self.sigmoid_derivative


    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def leaky_relu(self, x):
        return np.maximum(0.1 * x, x)

    def leaky_relu_derivative(self, x):
        return np.where(x > 0, 1, 0.1)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.activation_function(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.activation_function(self.z2)

        pi = self.softmax(np.dot(self.a2, self.W_pi) + self.b_pi)
        mu = np.dot(self.a2, self.W_mu) + self.b_mu
        sigma = np.exp(np.dot(self.a2, self.W_sigma) + self.b_sigma)

        return pi, mu, sigma
    
    def backprop(self, x, y, pi, mu, sigma, learning_rate=0.001):
        n = y.shape[0]
        pi_grad = np.zeros_like(pi)
        mu_grad = np.zeros_like(mu)
        sigma_grad = np.zeros_like(sigma)

        for i in range(n):
            for j in range(pi.shape[1]):
                coeff = pi[i, j]
                diff = (y[i] - mu[i, j])
                exponent = -0.5 * (diff / sigma[i, j]) ** 2
                normalizer = (1 / (np.sqrt(2 * np.pi) * sigma[i, j]))
                likelihood = normalizer * np.exp(exponent)
                responsibility = coeff * likelihood / (np.sum(pi[i] * normalizer * np.exp(-0.5 * ((y[i] - mu[i]) / sigma[i]) ** 2)) + 1e-8)

                pi_grad[i, j] = responsibility - coeff
                mu_grad[i, j] = responsibility * diff / (sigma[i, j] ** 2)
                sigma_grad[i, j] = responsibility * ((diff ** 2) / (sigma[i, j] ** 3) - 1 / sigma[i, j])

        dW_pi = np.dot(self.a2.T, pi_grad)
        db_pi = np.sum(pi_grad, axis=0, keepdims=True)

        dW_mu = np.dot(self.a2.T, mu_grad)
        db_mu = np.sum(mu_grad, axis=0, keepdims=True)

        dW_sigma = np.dot(self.a2.T, sigma_grad)
        db_sigma = np.sum(sigma_grad, axis=0, keepdims=True)

        da2 = (np.dot(pi_grad, self.W_pi.T) +
               np.dot(mu_grad, self.W_mu.T) +
               np.dot(sigma_grad, self.W_sigma.T))
        dz2 = da2 * self.derivative_activation(self.z2)

        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.derivative_activation(self.z1) 

        dW1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W_pi -= learning_rate * dW_pi
        self.b_pi -= learning_rate * db_pi

        self.W_mu -= learning_rate * dW_mu
        self.b_mu -= learning_rate * db_mu

        self.W_sigma -= learning_rate * dW_sigma
        self.b_sigma -= learning_rate * db_sigma

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, x_train, y_train, epochs=1000, learning_rate=0.001):
        for epoch in range(epochs):
            # Forward pass
            pi, mu, sigma = self.forward(x_train)
            
            # Compute loss (negative log likelihood)
            n = y_train.shape[0]
            loss = 0
            for i in range(n):
                mixture_likelihood = 0
                for j in range(pi.shape[1]):
                    coeff = pi[i, j]
                    likelihood = (1 / (np.sqrt(2 * np.pi) * sigma[i, j])) * np.exp(-0.5 * ((y_train[i] - mu[i, j]) / sigma[i, j])**2)
                    mixture_likelihood += coeff * likelihood
                loss += -np.log(mixture_likelihood + 1e-8)
            loss /= n
            
            # Print loss every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

            # backprop pass
            self.backprop(x_train, y_train, pi, mu, sigma, learning_rate=learning_rate)
