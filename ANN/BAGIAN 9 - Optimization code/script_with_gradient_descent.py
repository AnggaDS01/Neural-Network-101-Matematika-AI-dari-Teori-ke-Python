import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from NeuralNet import Linear, MeanSquaredError

# Generate dummy data
seed = 42
np.random.seed(seed)

# Membuat data dummy
luas_rumah = np.round(np.random.uniform(50, 400, 500).reshape(-1, 1), 2)
harga_rumah = np.round(
    (luas_rumah * 5 + np.random.normal(0, 75, luas_rumah.shape)).reshape(-1, 1), 2
)
X = luas_rumah
Y = harga_rumah

# Inisialisasi model
input_to_output_layer = Linear(in_features=1, out_features=1)
loss = MeanSquaredError()

# Hyperparameter
learning_rate = 0.00001 # Disesuaikan agar konvergensi stabil
iterations = 5000
stagnation_counter = 0
best_loss = float("inf")
patience = 5  # Jumlah iterasi yang diperbolehkan tanpa penurunan loss
epsilon = 1e-3 # Threshold minimum perubahan loss yang dianggap signifikan

# Menyimpan histori loss dan prediksi
loss_history = []
prediction_history = []
iteration_history = []

# Training loop menggunakan Gradient Descent
for i in range(iterations):
    # Forward pass
    Y_pred = input_to_output_layer.forward(X)

    # Hitung loss
    mse_loss = loss.calculate(Y, Y_pred)
    loss_history.append(mse_loss)

    # Hitung gradien menggunakan turunan Mean Squared Error
    dL_dY_pred = (-2 / len(Y)) * (Y - Y_pred)  # dL/dY_pred
    dL_dW = np.sum(dL_dY_pred * X)  # dL/dW
    dL_dB = np.sum(dL_dY_pred)  # dL/dB

    # Update bobot dan bias dengan Gradient Descent
    input_to_output_layer.weight -= learning_rate * dL_dW
    input_to_output_layer.bias -= learning_rate * dL_dB

    # Early stopping condition dengan threshold
    if best_loss - mse_loss > epsilon:  # Jika penurunan loss lebih besar dari threshold
        print(f"Iterasi {i}: mse_loss turun dari {best_loss:.6f} ke {mse_loss:.6f}")
        best_loss = mse_loss
        stagnation_counter = 0  # Reset counter jika loss turun signifikan
    else:
        stagnation_counter += 1  # Tambah counter jika loss stagnan
        print(f"Iterasi {i}: Loss stagnan ({stagnation_counter}/{patience})")

    # Simpan prediksi untuk animasi hanya jika loss turun atau belum mencapai patience
    if stagnation_counter < patience:
        prediction_history.append(Y_pred.flatten())
        iteration_history.append(i)

    # Jika stagnasi lebih lama dari patience, hentikan loop
    if stagnation_counter >= patience:
        print(f"Training dihentikan lebih awal pada iterasi {i} karena loss tidak turun signifikan selama {patience} iterasi.")
        break


# Fungsi update untuk animasi
def update_plot(frame):
    plt.cla()
    plt.scatter(X, Y, color="blue", label="Data Asli")
    plt.plot(X, prediction_history[frame], color="red", label="Prediksi")
    plt.xlabel("Luas Rumah (m^2)")
    plt.ylabel("Harga Rumah (juta rupiah)")
    plt.title(
        f"Iterasi {iteration_history[frame]}, Loss: {loss_history[iteration_history[frame]]:.2f}"
    )
    plt.legend()
    plt.grid(True)


# Membuat animasi
fig, ax = plt.subplots()
ani = FuncAnimation(
    fig, update_plot, frames=len(prediction_history), interval=100, repeat=False
)
plt.show()