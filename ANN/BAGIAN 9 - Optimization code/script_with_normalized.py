import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from NeuralNet import Linear, MeanSquaredError, SumSquaredError

# Generate dummy data
np.random.seed(42)

# Luas rumah (m^2), antara 50 hingga 300 meter persegi
luas_rumah = np.round(np.random.uniform(50, 400, 500).reshape(-1, 1), 2)

# Harga rumah (dalam juta rupiah), dengan variasi acak
harga_rumah = np.round(
    (luas_rumah * 5 + np.random.normal(0, 75, luas_rumah.shape)).reshape(-1, 1), 2
)

X = luas_rumah
Y = harga_rumah


# Normalisasi data
def minmaxscaler(data):
    return (data - data.min()) / (data.max() - data.min())


X_scaled = minmaxscaler(X)

# Inisialisasi layer-layer
# Layer 1: 1 input -> 2 output
input_to_hidden_layer = Linear(in_features=1, out_features=2)
# Layer 2: 2 input -> 3 output
hidden_to_hidden_layer = Linear(in_features=2, out_features=3)
# Layer 3: 3 input -> 1 output
hidden_to_output_layer = Linear(in_features=3, out_features=1)

# Inisialisasi Mean Squared Error sebagai loss function
loss = MeanSquaredError()
lowest_loss = np.inf  # some initial value

best_HL_1_weights, best_HL_1_biases = (
    input_to_hidden_layer.weight.copy(),
    input_to_hidden_layer.bias.copy(),
)
best_HL_2_weights, best_HL_2_biases = (
    hidden_to_hidden_layer.weight.copy(),
    hidden_to_hidden_layer.bias.copy(),
)
best_output_layer_weights, best_output_layer_biases = (
    hidden_to_output_layer.weight.copy(),
    hidden_to_output_layer.bias.copy(),
)

# List untuk menyimpan prediksi dan iterasi
prediction_history = []
iteration_history = []


# Fungsi untuk update plot
def update_plot(frame):
    plt.cla()
    plt.scatter(X_scaled, Y, color="blue", label="Data Asli")  # Plot data asli
    plt.plot(
        X_scaled, prediction_history[frame], color="red", label="Prediksi"
    )  # Plot prediksi
    plt.xlabel("Luas Rumah (m^2)")
    plt.ylabel("Harga Rumah (juta rupiah)")
    plt.title(f"Iterasi {iteration_history[frame]}, Loss: {loss_history[frame]:.2f}")
    plt.legend()
    plt.grid(True)


# Loop training
loss_history = []  # Untuk menyimpan nilai loss
for i in range(5000):
    # Update weights and biases di hidden layer 1
    input_to_hidden_layer.weight -= np.random.normal(
        0, 0.1, input_to_hidden_layer.weight.shape
    )
    input_to_hidden_layer.bias -= np.random.normal(
        0, 0.1, input_to_hidden_layer.bias.shape
    )

    # Update weights and biases di hidden layer 2
    hidden_to_hidden_layer.weight -= np.random.normal(
        0, 0.1, hidden_to_hidden_layer.weight.shape
    )
    hidden_to_hidden_layer.bias -= np.random.normal(
        0, 0.1, hidden_to_hidden_layer.bias.shape
    )

    # Update weights and biases di output layer
    hidden_to_output_layer.weight -= np.random.normal(
        0, 0.1, hidden_to_output_layer.weight.shape
    )
    hidden_to_output_layer.bias -= np.random.normal(
        0, 0.1, hidden_to_output_layer.bias.shape
    )

    # Forward pass melalui jaringan
    Z_1 = input_to_hidden_layer.forward(X_scaled)  # Output dari layer 1
    Z_2 = hidden_to_hidden_layer.forward(Z_1)  # Output dari layer 2
    Y_pred = hidden_to_output_layer.forward(Z_2)  # Output akhir

    # Hitung rata-rata dari kuadrat selisih
    mse_loss = loss.calculate(Y, Y_pred)

    if mse_loss < lowest_loss:
        # print(f"nilai parameter terbaik ditemukan pada iterasi ke-{i}, dengan loss: {mse_loss}")
        best_HL_1_weights, best_HL_1_biases = (
            input_to_hidden_layer.weight.copy(),
            input_to_hidden_layer.bias.copy(),
        )
        best_HL_2_weights, best_HL_2_biases = (
            hidden_to_hidden_layer.weight.copy(),
            hidden_to_hidden_layer.bias.copy(),
        )
        best_output_layer_weights, best_output_layer_biases = (
            hidden_to_output_layer.weight.copy(),
            hidden_to_output_layer.bias.copy(),
        )
        lowest_loss = mse_loss

        # Simpan prediksi dan iterasi
        prediction_history.append(Y_pred.flatten())
        iteration_history.append(i)
        loss_history.append(mse_loss)
    else:
        input_to_hidden_layer.weight, input_to_hidden_layer.bias = (
            best_HL_1_weights.copy(),
            best_HL_1_biases.copy(),
        )
        hidden_to_hidden_layer.weight, hidden_to_hidden_layer.bias = (
            best_HL_2_weights.copy(),
            best_HL_2_biases.copy(),
        )
        hidden_to_output_layer.weight, hidden_to_output_layer.bias = (
            best_output_layer_weights.copy(),
            best_output_layer_biases.copy(),
        )

# Buat animasi
fig, ax = plt.subplots()
ani = FuncAnimation(
    fig, update_plot, frames=len(prediction_history), interval=100, repeat=False
)

plt.show()
