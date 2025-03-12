import numpy as np

class MeanSquaredError:
    """
    Kelas untuk menghitung Mean Squared Error (MSE) Loss.

    MSE Loss adalah metrik yang mengukur rata-rata kuadrat selisih antara nilai prediksi
    dan nilai target. Ini sering digunakan dalam masalah regresi untuk mengevaluasi
    seberapa baik model memprediksi nilai target.

    Rumus MSE:
        MSE = (1 / n) * Σ(prediksi - target)^2

    Atribut:
        None
    """

    def __init__(self):
        """
        Inisialisasi objek MeanSquaredError.

        Kenapa tidak ada atribut yang diinisialisasi? Karena MSE adalah fungsi stateless,
        artinya tidak perlu menyimpan state apa pun antara perhitungan.
        """
        pass

    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Menghitung MSE Loss berdasarkan prediksi dan target.

        Kenapa perlu menghitung MSE? Untuk mengevaluasi performa model dengan mengukur
        seberapa jauh prediksi menyimpang dari target.

        Parameter:
            y_pred (np.ndarray): Array prediksi dengan bentuk (batch_size, output_features).
            y_true (np.ndarray): Array target dengan bentuk (batch_size, output_features).

        Returns:
            float: Nilai MSE Loss.

        Raises:
            ValueError: Jika bentuk `y_pred` dan `y_true` tidak sama.
        """
        # Validasi bentuk input
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Bentuk prediksi {y_pred.shape} dan target {y_true.shape} tidak sesuai."
            )

        # Hitung selisih antara prediksi dan target
        # Kenapa perlu hitung selisih? Karena MSE mengukur seberapa jauh prediksi dari target.
        element_wise_error = y_pred - y_true

        # Kuadratkan selisih
        # Kenapa perlu kuadratkan? Untuk menghilangkan tanda negatif dan memberikan
        # penalti yang lebih besar untuk kesalahan yang besar.
        squared_error = np.square(element_wise_error)

        # Hitung rata-rata dari kuadrat selisih
        # Kenapa perlu rata-rata? Karena MSE adalah rata-rata kesalahan kuadrat
        # untuk semua sampel dalam batch.
        mse_loss = np.mean(squared_error)

        return mse_loss