import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
			layer ini melakukan transformasi linier:
			Z = X * W + B
			
			Parameters:
			- in_features: Jumlah fitur input (n_inputs)
			- out_features: Jumlah neuron di layer ini (n_neurons)
        """
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.randn(in_features, out_features) * 0.01  # (n_inputs, n_neurons)
        self.bias = np.zeros((1, out_features))  # (1, n_neurons)

    def forward(self, x):
        """
			Melakukan forward pass seperti torch.nn.Linear
			
			Parameters:
			- x: Input dengan shape (batch_size, in_features)
			
			Returns:
			- Output setelah transformasi linier dengan shape (batch_size, out_features)
        """
        return np.dot(x, self.weight) + self.bias # Z(m, n) = X(m, ℓ) * W(n, ℓ) + B(1, n)
    

X = np.array([
    [100, 3, 2],
    [150, 5, 3],
    [175, 5, 4],
    [200, 6, 3],
])

W = np.array([
    [0.1, 0.4],
    [0.2, 0.5],
    [0.3, 0.6],
])

B = np.array([0.0, 0.0])

Y = np.array([
    [500], 
    [850], 
    [900], 
    [1100]
])