import numpy as np
from NeuralNet import Linear, MeanSquaredError

# Set seed untuk reproducibility
np.random.seed(1)

X = np.array([
	[100, 3, 2],
	[150, 5, 3],
	[175, 5, 4],
	[200, 6, 3],
])

Y = np.array([
	[500],
	[850],
	[900],
	[1100],
])

# Inisialisasi layer-layer
# Layer 1: 3 input -> 2 output
input_to_hidden_layer = Linear(in_features=3, out_features=2)  
# Layer 2: 2 input -> 3 output
hidden_to_hidden_layer = Linear(in_features=2, out_features=3)
# Layer 3: 3 input -> 1 output  
hidden_to_output_layer = Linear(in_features=3, out_features=1)

# Forward pass melalui jaringan
Z_1 = input_to_hidden_layer.forward(X)  # Output dari layer 1
Z_2 = hidden_to_hidden_layer.forward(Z_1)  # Output dari layer 2
Y_pred = hidden_to_output_layer.forward(Z_2)  # Output akhir

# Cetak hasil prediksi
print("Final output shape:", Y_pred.shape)
print("Final output:\n", Y_pred)

# Hitung rata-rata dari kuadrat selisih
loss = MeanSquaredError()
mse_loss = loss.calculate(Y, Y_pred)
print("\nMSE Loss:", mse_loss)