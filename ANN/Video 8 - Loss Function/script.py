import numpy as np
from NeuralNet import Linear, MeanSquaredError  # Import kelas Linear dan MeanSquaredError

# Set seed untuk reproducibility
# Kenapa pake np.random.seed? Biar hasil random selalu konsisten setiap kali di-run.
np.random.seed(1)

# Input matrix X
# Kenapa bentuknya (4 x 3)? Karena ada 4 sampel dan 3 fitur.
X = np.array([
    [100, 3, 2],
    [150, 5, 3],
    [175, 5, 4],
    [200, 6, 3],
])

# Target matrix Y
# Kenapa bentuknya (4 x 1)? Karena ada 4 sampel dan 1 target per sampel.
Y = np.array([
    [500],
    [850],
    [900],
    [1100],
])

# Inisialisasi layer-layer
# Layer 1: Input ke Hidden Layer
# Kenapa in_features=3 dan out_features=2? Karena input punya 3 fitur, dan kita mau 2 neuron di hidden layer.
input_to_hidden_layer = Linear(in_features=3, out_features=2)

# Layer 2: Hidden Layer ke Hidden Layer
# Kenapa in_features=2 dan out_features=3? Karena output dari layer sebelumnya punya 2 neuron, dan kita mau 3 neuron di layer ini.
hidden_to_hidden_layer = Linear(in_features=2, out_features=3)

# Layer 3: Hidden Layer ke Output Layer
# Kenapa in_features=3 dan out_features=1? Karena output dari layer sebelumnya punya 3 neuron, dan kita mau 1 output (misal untuk regresi).
hidden_to_output_layer = Linear(in_features=3, out_features=1)

# Forward pass melalui jaringan
# Kenapa perlu forward pass? Untuk menghitung output dari setiap layer berdasarkan input, weight, dan bias.
Z_1 = input_to_hidden_layer.forward(X)  # Output dari layer 1
Z_2 = hidden_to_hidden_layer.forward(Z_1)  # Output dari layer 2
Y_pred = hidden_to_output_layer.forward(Z_2)  # Output akhir

# Cetak hasil prediksi
# Kenapa perlu print shape? Untuk memastikan output punya bentuk yang sesuai dengan ekspektasi.
print("Final output shape:", Y_pred.shape)
print("Final output:\n", Y_pred)

# Hitung MSE Loss
# Kenapa perlu hitung MSE? Untuk mengevaluasi seberapa jauh prediksi menyimpang dari target.
loss = MeanSquaredError()
mse_loss = loss.calculate(Y, Y_pred)
print("\nMSE Loss:", mse_loss)