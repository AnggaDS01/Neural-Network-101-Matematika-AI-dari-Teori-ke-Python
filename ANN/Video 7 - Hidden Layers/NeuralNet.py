import numpy as np


class Linear:
    def __init__(self, in_features: int, out_features: int) -> None:
        """
        Inisialisasi layer linear (fully connected layer) dalam neural network.

        Kenapa pake layer linear? Karena ini adalah blok dasar untuk transformasi linear
        dalam neural network, di mana input di-transform menggunakan weight dan bias.

        Args:
            in_features (int): Jumlah fitur input (ℓ)
            out_features (int): Jumlah neuron dalam layer (n).
        """
        self.in_features = in_features  # Jumlah fitur input (ℓ)
        self.out_features = out_features  # Jumlah neuron dalam layer (n)

        # Inisialisasi weight dengan nilai random kecil
        # Kenapa pake random kecil? Untuk menghindari masalah seperti vanishing/exploding gradients
        # saat training neural network.
        self.weight = (
            np.random.randn(in_features, out_features) * 0.01
        )  # Matriks weight (ℓ x n)

        # Inisialisasi bias dengan nilai 0
        # Kenapa bias diinisialisasi dengan 0? Karena bias biasanya dimulai dari 0 dan
        # akan di-update selama proses training.
        self.bias = np.zeros((1, out_features))  # Vektor bias (1 x n)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Melakukan forward pass (transformasi linear) pada input x.

        Kenapa perlu forward pass? Karena ini adalah langkah untuk menghitung output
        dari layer linear berdasarkan input, weight, dan bias.

        Args:
            x (np.ndarray): Input matrix berukuran (m x ℓ), di mana:
                            - m = jumlah sampel
                            - ℓ = jumlah fitur input

        Returns:
            np.ndarray: Output matrix berukuran (m x n), di mana:
                        - m = jumlah sampel
                        - n = jumlah neuron dalam layer

        Raises:
            ValueError: Jika bentuk input tidak sesuai dengan in_features.
        """
        # Validasi bentuk input
        if x.shape[1] != self.in_features:
            raise ValueError(
                f"Bentuk input {x.shape} tidak sesuai dengan in_features {self.in_features}"
            )

        # Transformasi linear: Z = X * W + B
        # Kenapa pake np.dot? Karena np.dot adalah fungsi NumPy untuk perkalian matriks,
        # yang dioptimalkan untuk kecepatan dan efisiensi.
        return (
            np.dot(x, self.weight) + self.bias
        )  # Z(m, n) = X(m, ℓ) * W(ℓ, n) + B(1, n)
