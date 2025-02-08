# Implementasi manual tanpa library
def forward_pass(X, W, B):
    """
    Melakukan forward pass sederhana tanpa library eksternal.

    Args:
    X : List of List (m x l) - Input matrix (m samples, l features)
    W : List of List (l x n) - Weight matrix (l features, n output neurons)
    B : List (n) - Bias vector

    Returns:
    Z : List of List (m x n) - Output after applying linear transformation
    """
    m = len(X)     # Jumlah sampel
    l = len(X[0])  # Jumlah inputs
    n = len(B)  # Jumlah neuron di layer berikutnya

    # Inisialisasi output matrix Z dengan ukuran (m x n)
    Z = [[0] * n for _ in range(m)]

    # Perkalian matriks manual: Z = X * W + B
    for i in range(m):      # Iterasi tiap sampel
        for j in range(n):  # Iterasi tiap neuron di layer berikutnya
            Z[i][j] = B[j]  # Mulai dari bias
            for k in range(l):  # Iterasi tiap inputs untuk dot product
                Z[i][j] = X[i][k] * W[k][j] + Z[i][j]

    return Z