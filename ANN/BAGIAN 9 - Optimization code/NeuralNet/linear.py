import numpy as np


class Linear:
    """
    Lapisan linear yang melakukan transformasi linear:
    Z = X * W + B

    Atribut:
        in_features (int): Jumlah fitur input.
        out_features (int): Jumlah neuron dalam lapisan.
        weight (np.ndarray): Matriks bobot dengan bentuk (in_features, out_features).
        bias (np.ndarray): Vektor bias dengan bentuk (1, out_features).
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """
        Inisialisasi lapisan Linear.

        Parameter:
            in_features (int): Jumlah fitur input.
            out_features (int): Jumlah neuron dalam lapisan.
        """
        self.in_features = in_features  # Jumlah fitur input
        self.out_features = out_features  # Jumlah neuron dalam lapisan
        self.weight = (
            np.random.randn(in_features, out_features) * 0.01
        )  # (n_inputs, n_neurons)
        self.bias = np.zeros((1, out_features))  # (1, n_neurons)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Melakukan forward pass dari transformasi linear.

        Parameter:
            x (np.ndarray): Array input dengan bentuk (batch_size, in_features) atau (samples, in_features).

        Returns:
            np.ndarray: Array output dengan bentuk (batch_size, out_features).

        Raises:
            ValueError: Jika array input tidak memiliki bentuk yang sesuai.
        """
        if x.shape[1] != self.in_features:
            raise ValueError(
                f"Bentuk input {x.shape} tidak sesuai dengan in_features {self.in_features}"
            )

        return (
            np.dot(x, self.weight) + self.bias
        )  # Z(m, n) = X(m, ℓ) * W(ℓ, n) + B(1, n)