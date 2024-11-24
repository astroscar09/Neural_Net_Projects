import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
class Optimizer_SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

class Optimizer_Adagrad:
    def __init__(self, learning_rate=1.0, decay=1e-7):
        self.learning_rate = learning_rate
        self.decay = decay
        self.cache = None
    def update_params(self, layer):
        if self.cache is None:
            self.cache = np.zeros_like(layer.weights)
        self.cache += layer.dweights ** 2
        layer.weights += -self.learning_rate * layer.dweights / (np.sqrt(self.cache) + self.decay)
        layer.biases += -self.learning_rate * layer.dbiases

class Optimizer_RMSprop:
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.cache = None
    def update_params(self, layer):
        if self.cache is None:
            self.cache = np.zeros_like(layer.weights)
        self.cache = self.decay * self.cache + (1 - self.decay) * layer.dweights ** 2
        layer.weights += -self.learning_rate * layer.dweights / (np.sqrt(self.cache) + self.epsilon)
        layer.biases += -self.learning_rate * layer.dbiases

class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay1=0.9, decay2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.decay1 = decay1
        self.decay2 = decay2
        self.epsilon = epsilon
        self.iterations = 0
        self.m = None
        self.v = None
    def update_params(self, layer):
        if self.m is None:
            self.m = np.zeros_like(layer.weights)
            self.v = np.zeros_like(layer.weights)
        self.iterations += 1
        self.m = self.decay1 * self.m + (1 - self.decay1) * layer.dweights
        self.v = self.decay2 * self.v + (1 - self.decay2) * layer.dweights ** 2
        m_corrected = self.m / (1 - self.decay1 ** self.iterations)
        v_corrected = self.v / (1 - self.decay2 ** self.iterations)
        layer.weights += -self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
        layer.biases += -self.learning_rate * layer.dbiases

class Layer_Input:
    def forward(self, inputs, training):
        self.output = inputs

class Model:
    def __init__(self):
        self.layers = []
    def add(self, layer):
        self.layers.append(layer)
    def set(self, *, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer
    def finalize(self):
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)
        self.trainable_layers = []
        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.loss.layer = self.layers[i]
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
        self.loss.remember_trainable_layers(self.trainable_layers)
        if isinstance(self.optimizer, Optimizer_Adam):
            self.iterations = 0
    def forward(self, X, training):
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        return layer.output
    def backward(self, output, y):
        if self.loss is not None:
            self.loss.backward(output, y)
            for layer in reversed(self.layers):
                layer.backward(layer.next.dvalues)
    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        self.accuracy = {'train': [], 'validation': []}
        self.losses = {'train': [], 'validation': []}
        if validation_data is not None:
            X_val, y_val = validation_data
        for epoch in range(1, epochs + 1):
            output = self.forward(X, training=True)
            data_loss = self.loss.calculate(output, y)
            loss = data_loss
            self.losses['train'].append(loss)
            accuracy = self.accuracy_calculation(output, y)
            self.accuracy['train'].append(accuracy)
            self.backward(output, y)
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update