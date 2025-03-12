import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.animation import FuncAnimation
from NeuralNet import Linear, MeanSquaredError



def train_model(
        seed: int = 42,
        n_samples: int = 500,
        n_features: int = 1,
        iterations: int = 5000,
        file_save: str = 'training_random_search_results.pkl'
    ) -> dict:

    np.random.seed(seed)

    # membuat data dummy
    # Luas rumah (m^2), antara 50 hingga 400 meter persegi sebanyak 500 samples
    luas_rumah = np.round(np.random.uniform(50, 400, n_samples).reshape(-1, 1), 2)

    # Harga rumah (dalam juta rupiah), dengan variasi acak
    harga_rumah = np.round(
        (luas_rumah * 5 + np.random.normal(0, 75, luas_rumah.shape)).reshape(-1, 1), 2
    )

    X = luas_rumah
    Y = harga_rumah

    # Inisialisasi layer-layer
    # Layer 1: 1 input -> 2 output
    input_to_output_layer = Linear(in_features=n_features, out_features=1)

    # Inisialisasi Mean Squared Error sebagai loss function
    loss = MeanSquaredError()
    lowest_loss = float("inf")  # some initial value

    best_HL_1_weights, best_HL_1_biases = (
        input_to_output_layer.weight.copy(),
        input_to_output_layer.bias.copy(),
    )

    # List untuk menyimpan prediksi dan iterasi
    loss_history = []
    weight_history = []
    bias_history = []
    iteration_history = []

    # Training loop menggunakan Random Search
    print("Mulai pelatihan model...")
    for i in range(iterations):
        # Update weights and biases di hidden layer 1
        input_to_output_layer.weight -= np.random.normal(
            0, 0.1, input_to_output_layer.weight.shape
        )
        input_to_output_layer.bias -= np.random.normal(
            0, 0.1, input_to_output_layer.bias.shape
        )

        # Forward pass melalui jaringan
        Y_pred = input_to_output_layer.forward(X)  # Output akhir

        # Hitung rata-rata dari kuadrat selisih
        mse_loss = loss.calculate(Y, Y_pred)

        if mse_loss < lowest_loss:
            print(f'nilai parameter terbaik ditemukan pada iterasi ke-{i}, dengan loss: {lowest_loss}')
            best_HL_1_weights, best_HL_1_biases = (
                input_to_output_layer.weight.copy(),
                input_to_output_layer.bias.copy(),
            )

            lowest_loss = mse_loss
            # Simpan weight dan bias saat ini
            weight_history.append(input_to_output_layer.weight.item())
            bias_history.append(input_to_output_layer.bias.item())
            loss_history.append(mse_loss)
            iteration_history.append(i)
        else:
            input_to_output_layer.weight, input_to_output_layer.bias = (
                best_HL_1_weights.copy(),
                best_HL_1_biases.copy(),
            )

    # Plot loss history    # Simpan hasil training ke file
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