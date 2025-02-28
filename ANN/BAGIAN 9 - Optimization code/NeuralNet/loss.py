import numpy as np


class MeanSquaredError:
    """
    Kelas untuk menghitung Mean Squared Error (MSE) Loss.

    MSE Loss mengukur rata-rata kuadrat selisih antara nilai prediksi dan nilai target.
    Rumus MSE:
        MSE = (1 / n) * Σ(prediksi - target)^2

    Atribut:
        None
    """

    def __init__(self):
        """
        Inisialisasi objek MeanSquaredError.
        """
        pass

    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Menghitung MSE Loss berdasarkan prediksi dan target.

        Parameter:
            y_pred (np.ndarray): Array prediksi dengan bentuk (batch_size, output_features).
            y_true (np.ndarray): Array target dengan bentuk (batch_size, output_features).

        Returns:
            float: Nilai MSE Loss.

        Raises:
            ValueError: Jika bentuk `y_pred` dan `y_true` tidak sama.
        """
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Bentuk prediksi {y_pred.shape} dan target {y_true.shape} tidak sesuai."
            )

        # Hitung selisih antara prediksi dan target
        element_wise_error = y_pred - y_true
        # Kuadratkan selisih
        squared_error = np.square(element_wise_error)
        # Hitung rata-rata dari kuadrat selisih
        mse_loss = np.mean(squared_error)

        return mse_loss


class SumSquaredError:
    """
    Kelas untuk menghitung Sum of Squared Error (SSE) Loss.

    SSE Loss mengukur total kuadrat selisih antara nilai prediksi dan nilai target.
    Rumus SSE:
        SSE = Σ(prediksi - target)^2

    Atribut:
        None
    """

    def __init__(self):
        """
        Inisialisasi objek SumSquaredError.
        """
        pass

    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Menghitung SSE Loss berdasarkan prediksi dan target.

        Parameter:
            y_pred (np.ndarray): Array prediksi dengan bentuk (batch_size, output_features).
            y_true (np.ndarray): Array target dengan bentuk (batch_size, output_features).

        Returns:
            float: Nilai SSE Loss.

        Raises:
            ValueError: Jika bentuk `y_pred` dan `y_true` tidak sama.
        """
        # Validasi bentuk array prediksi dan target
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Bentuk prediksi {y_pred.shape} dan target {y_true.shape} tidak sesuai."
            )

        # Hitung selisih antara prediksi dan target
        element_wise_error = y_pred - y_true
        # Kuadratkan selisih
        squared_error = np.square(element_wise_error)
        # Hitung jumlah dari kuadrat selisih (SSE)
        sse_loss = np.sum(squared_error)

        return sse_loss
