# Implementasi manual tanpa library
def forward_pass(
    X: list, 
    W: list, 
    B: list, 
    ) -> list:

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
    n = len(B)  # Jumlah neuron pada layer

    # Inisialisasi output matrix Z dengan ukuran (m x n)
    Z = [[0] * n for _ in range(m)]

    # Perkalian matriks manual: Z = X * W + B
    for i in range(m):      # Loop pertama: pilih baris data mana yang sedang diproses
        for j in range(n):  # Loop kedua: kita mau mengisi hasil untuk neuron mana
            Z[i][j] = B[j]  # Pertama, masukkan nilai bias dulu ∑x*w + b == b + ∑x*w
            for k in range(l):  # Loop ketiga: iterasi tiap inputs untuk dot product
                Z[i][j] += X[i][k] * W[k][j]

    return Z