import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


def calculate_loss_surface(weight_range, bias_range, X, Y):
    """Menghitung loss surface untuk berbagai nilai weight dan bias"""
    loss_surface = np.zeros((len(weight_range), len(bias_range)))
    for i, w in enumerate(weight_range):
        for j, b in enumerate(bias_range):
            y_pred = X * w + b
            loss_surface[i, j] = np.mean((Y - y_pred) ** 2)
    return loss_surface


# file_load: str = "training_gradient_descent_results.pkl"
# file_load: str = "training_random_search_results.pkl"


def visualize_gradient_descent(file_load: str = "training_gradient_descent_results.pkl"):
    # Cek apakah file hasil training ada
    if not os.path.exists(f"ANN/BAGIAN 9 - Optimization code/{file_load}"):
        print(f"File 'ANN/BAGIAN 9 - Optimization code/{file_load}' tidak ditemukan.")
        print(
            "Jalankan 'script_with_gradient_descent.py atau script_with_random_search.py' terlebih dahulu untuk melatih model."
        )
        return

    # Load hasil training
    print("Memuat hasil training...")
    with open(f"ANN/BAGIAN 9 - Optimization code/{file_load}", "rb") as f:
        results = pickle.load(f)

    X = results["X"]
    Y = results["Y"]
    weight_history = results["weight_history"]
    bias_history = results["bias_history"]
    loss_history = results["loss_history"]
    iteration_history = results["iteration_history"]

    print("Data berhasil dimuat.")
    print(f"Jumlah iterasi: {len(iteration_history)}")
    print(f"Weight akhir: {weight_history[-1]:.6f}")
    print(f"Bias akhir: {bias_history[-1]:.6f}")
    print(f"Loss akhir: {loss_history[-1]:.6f}")

    # Persiapkan visualisasi
    print("Menyiapkan visualisasi...")
    plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(15, 10))

    # Create loss surface for visualization
    # Define ranges for weight and bias
    w_min, w_max = min(weight_history) - 0.5, max(weight_history) + 0.5
    b_min, b_max = min(bias_history) - 25, max(bias_history) + 25
    weight_range = np.linspace(w_min, w_max, 30)  # Mengurangi resolusi agar lebih cepat
    bias_range = np.linspace(b_min, b_max, 30)
    W, B = np.meshgrid(weight_range, bias_range)

    print("Menghitung loss surface...")
    loss_surface = calculate_loss_surface(weight_range, bias_range, X, Y)
    print("Loss surface selesai dihitung.")

    # Buat sampel data dari history untuk animasi
    sample_rate = max(1, len(weight_history) // 100)  # Sampel 100 titik untuk animasi
    sampled_indices = range(0, len(weight_history), sample_rate)
    sampled_weights = [weight_history[i] for i in sampled_indices]
    sampled_biases = [bias_history[i] for i in sampled_indices]
    sampled_losses = [loss_history[i] for i in sampled_indices]
    sampled_iterations = [iteration_history[i] for i in sampled_indices]

    print(f"Membuat animasi dengan {len(sampled_weights)} frame...")

    # Fungsi inisialisasi untuk animasi
    def init():
        return update(0)

    # Update function for animation
    def update(frame):
        plt.clf()  # Clear the figure

        # First subplot: 3D surface plot
        ax1 = fig.add_subplot(221, projection="3d")
        surf = ax1.plot_surface(
            W, B, loss_surface, cmap=cm.coolwarm, alpha=0.6, linewidth=0
        )

        # Plot the current position
        current_w = sampled_weights[frame]
        current_b = sampled_biases[frame]
        current_loss = sampled_losses[frame]
        current_iter = sampled_iterations[frame]

        # Find loss value at current position
        w_idx = np.abs(weight_range - current_w).argmin()
        b_idx = np.abs(bias_range - current_b).argmin()
        try:
            z_val = loss_surface[
                b_idx, w_idx
            ]  # Note: meshgrid returns with swapped indices
        except:
            z_val = current_loss  # Fallback if indices are out of range

        ax1.scatter([current_w], [current_b], [z_val], color="black", s=100, marker="o")

        # Plot the trajectory up to the current point
        if frame > 0:
            trajectory_w = sampled_weights[: frame + 1]
            trajectory_b = sampled_biases[: frame + 1]
            trajectory_z = []

            for w, b in zip(trajectory_w, trajectory_b):
                w_idx = np.abs(weight_range - w).argmin()
                b_idx = np.abs(bias_range - b).argmin()
                try:
                    z = loss_surface[b_idx, w_idx]
                except:
                    z = current_loss
                trajectory_z.append(z)

            ax1.plot(trajectory_w, trajectory_b, trajectory_z, "r-", linewidth=2)

        ax1.set_xlabel("Weight")
        ax1.set_ylabel("Bias")
        ax1.set_zlabel("Loss")
        ax1.set_title(
            f"Loss Surface - Iterasi {current_iter}, Loss: {current_loss:.2f}"
        )

        # Second subplot: Contour plot (top view)
        ax2 = fig.add_subplot(222)
        contour = ax2.contourf(W, B, loss_surface, levels=20, cmap=cm.coolwarm)
        ax2.scatter([current_w], [current_b], color="black", s=100, marker="x")
        if frame > 0:
            ax2.plot(
                sampled_weights[: frame + 1],
                sampled_biases[: frame + 1],
                "r-o",
                linewidth=2,
                markersize=3,
            )
        ax2.set_xlabel("Weight")
        ax2.set_ylabel("Bias")
        ax2.set_title("Kontur Loss Function")
        plt.colorbar(contour, ax=ax2, label="Loss")

        # Third subplot: Loss vs Iterations
        ax3 = fig.add_subplot(223)
        ax3.plot(sampled_iterations[: frame + 1], sampled_losses[: frame + 1], "b-")
        ax3.scatter([current_iter], [current_loss], color="red", s=100)
        ax3.set_xlabel("Iterasi")
        ax3.set_ylabel("Loss")
        ax3.set_title("Loss vs Iterasi")
        ax3.grid(True)

        # Fourth subplot: Prediction vs Data
        ax4 = fig.add_subplot(224)
        ax4.scatter(X, Y, color="blue", alpha=0.3, label="Data Asli")

        # Current prediction line
        Y_pred = X * current_w + current_b
        sorted_indices = np.argsort(X.flatten())
        ax4.plot(
            X[sorted_indices],
            Y_pred[sorted_indices],
            color="red",
            linewidth=2,
            label="Prediksi",
        )
        ax4.set_xlabel("Luas Rumah (m^2)")
        ax4.set_ylabel("Harga Rumah (juta rupiah)")
        ax4.set_title(f"Prediksi dengan W={current_w:.4f}, B={current_b:.4f}")
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        return fig.get_axes()

    print("Memulai animasi...")
    # Create animation
    ani = FuncAnimation(
        fig,
        update,
        frames=len(sampled_weights),
        init_func=init,
        interval=100,
        blit=False,
        repeat=False,
    )

    plt.tight_layout()
    plt.show()

    # Pastikan window matplotlib tetap terbuka
    print("Animasi selesai. Tutup window plot untuk keluar program.")
    plt.ioff()  # Turn off interactive mode
    plt.show(block=True)  # Block execution until window is closed


if __name__ == "__main__":
    visualize_gradient_descent()
