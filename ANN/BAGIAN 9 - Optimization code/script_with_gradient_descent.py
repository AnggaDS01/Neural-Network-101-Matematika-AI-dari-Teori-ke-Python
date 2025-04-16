import numpy as np
import pickle
from NeuralNet import Linear, MeanSquaredError

def min_max_scaler(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def train_model(
        seed: int = 42,
        n_samples: int = 500,
        n_features: int = 1,
        iterations: int = 5000,
        file_save: str = 'training_gradient_descent_results.pkl',
        
        # Hyperparameter
        learning_rate: float = 0.1,  # Disesuaikan agar konvergensi stabil
        patience: int = 5,  # Jumlah iterasi yang diperbolehkan tanpa penurunan loss
        epsilon: float = 1e-3  # Threshold minimum perubahan loss yang dianggap signifikan
    ) -> dict:

    # set seed
    np.random.seed(seed)

    # Membuat data dummy
    luas_rumah = np.round(np.random.uniform(50, 400, n_samples).reshape(-1, 1), 2)
    harga_rumah = np.round(
        (luas_rumah * 5 + np.random.normal(0, 75, luas_rumah.shape)).reshape(-1, 1), 2
    )
    X = min_max_scaler(luas_rumah)
    Y = harga_rumah

    # Inisialisasi model
    input_to_output_layer = Linear(in_features=n_features, out_features=1)
    loss = MeanSquaredError()
    
    # Hyperparameter
    stagnation_counter = 0
    best_loss= float("inf")

    # Menyimpan histori loss, weight, bias dan prediksi
    loss_history = []
    weight_history = []
    bias_history = []
    iteration_history = []

    # Training loop menggunakan Gradient Descent
    print("Mulai pelatihan model...")
    for i in range(iterations):
        # Forward pass
        Y_pred = input_to_output_layer.forward(X)
        
        # Simpan weight dan bias saat ini
        weight_history.append(input_to_output_layer.weight.item())
        bias_history.append(input_to_output_layer.bias.item())

        # Hitung loss
        mse_loss = loss.calculate(Y, Y_pred)
        loss_history.append(mse_loss)
        iteration_history.append(i)

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

        # Jika stagnasi lebih lama dari patience, hentikan loop
        if stagnation_counter >= patience:
            print(f"Training dihentikan lebih awal pada iterasi {i} karena loss tidak turun signifikan selama {patience} iterasi.")
            break

    print("Pelatihan selesai.")
    
    # Simpan hasil training ke file
    training_results = {
        'X': X,
        'Y': Y,
        'weight_history': weight_history,
        'bias_history': bias_history,
        'loss_history': loss_history,
        'iteration_history': iteration_history,
        'final_weight': input_to_output_layer.weight.item(),
        'final_bias': input_to_output_layer.bias.item(),
        'final_loss': mse_loss
    }
    
    with open(f'ANN/BAGIAN 9 - Optimization code/{file_save}', 'wb') as f:
        pickle.dump(training_results, f)
    
    print(f"Hasil training disimpan ke file '{file_save}'")
    return training_results

if __name__ == "__main__":
    results = train_model()
    
    # Tampilkan hasil akhir
    print("\nHasil Akhir Pelatihan:")
    print(f"Weight: {results['final_weight']:.6f}")
    print(f"Bias: {results['final_bias']:.6f}")
    print(f"Loss Akhir: {results['final_loss']:.6f}")
    print("\nAnda dapat menjalankan 'visualize_training.py' untuk melihat animasi gradient descent.")