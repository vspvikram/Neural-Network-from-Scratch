import numpy as np

# Adam optimizer
class Optimizer:
    def __init__(self, learning_rate=None, name=None):
        self.learning_rate = learning_rate
        self.name = name
    
    def config(self, layers):
        raise NotImplementedError

    def optimize(self, layers):
        raise NotImplementedError


class Adam(Optimizer):
    def __init__(self, learning_rate = 0.01, name = None, beta1 = 0.9, beta2=0.999,
                eps=1e-7):
        super(Adam, self).__init__(learning_rate, name)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # first and second momemt 
        self.m = {}
        self.v = {}

    def set_config(self, layers):
        for idx in layers.keys():
            if layers[idx].is_trainable:
                self.m[f"weights{idx}"] = 0
                self.m[f"biases{idx}"] = 0

                self.v[f"weights{idx}"] = 0
                self.v[f"biases{idx}"] = 0

    def optimize(self, layers, idx, step_number):
        assert step_number > 0, "Step number should be greater than 0"
        grad_weights = layers[idx].grad_weights
        grad_biases = layers[idx].grad_biases

        # calculating the moments for weights and biases
        self.m[f"weights{idx}"] = self.beta1 * self.m[f"weights{idx}"] + (1 - self.beta1) * grad_weights
        self.v[f"weights{idx}"] = self.beta2 * self.v[f"weights{idx}"] + (1-self.beta2)*(grad_weights**2)
        
        self.m[f"biases{idx}"] = self.beta1 * self.m[f"biases{idx}"] + (1 - self.beta1) * grad_biases
        self.v[f"biases{idx}"] = self.beta2 * self.v[f"biases{idx}"] + (1-self.beta2)*(grad_biases**2)

        # computing the bias corrected moments wrt time
        m_demon = 1 - self.beta1**step_number
        weight_mt = self.m[f"weights{idx}"] / m_demon
        bias_mt = self.m[f"biases{idx}"] / m_demon

        v_demon = 1 - self.beta2 ** step_number
        weight_vt = self.v[f"weights{idx}"] / v_demon
        bias_vt = self.v[f"biases{idx}"] / v_demon

        # updating the weights and biases
        layers[idx].weights -= self.learning_rate * weight_mt / (np.sqrt(weight_vt) + self.eps)
        layers[idx].biases -= self.learning_rate * bias_mt / (np.sqrt(bias_vt) + self.eps)
