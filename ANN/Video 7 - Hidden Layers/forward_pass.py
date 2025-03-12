import numpy as np

# Implementasi menggunakan NumPy
def forward_pass_numpy(X, W, B):
    """
    Melakukan forward pass menggunakan NumPy.

    Kenapa pake NumPy? Karena NumPy dioptimalkan untuk operasi matriks,
    sehingga lebih cepat dan efisien dibanding implementasi manual.

    Args:
        X (np.ndarray): Matriks input berukuran (m x l), di mana:
                         - m = jumlah sampel
                         - l = jumlah fitur
        W (np.ndarray): Matriks weight berukuran (l x n), di mana:
                         - l = jumlah fitur (harus sama dengan X)
                         - n = jumlah neuron di layer berikutnya
        B (np.ndarray): Vektor bias berukuran (n), di mana:
                         - n = jumlah neuron di layer berikutnya

    Returns:
        Z (np.ndarray): Matriks output berukuran (m x n), di mana:
                         - m = jumlah sampel
                         - n = jumlah neuron di layer berikutnya
    """
    return np.dot(X, W) + B  # NumPy melakukan perkalian matriks dan penambahan bias secara efisien