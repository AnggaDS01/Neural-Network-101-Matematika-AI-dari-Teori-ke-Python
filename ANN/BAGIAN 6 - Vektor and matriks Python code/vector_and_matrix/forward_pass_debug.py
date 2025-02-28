# Implementasi manual tanpa library
def forward_pass_debug(
        X: list, 
        W: list, 
        B: list, 
        debug: bool=False
    ) -> list:
    """
        Melakukan forward pass sederhana tanpa library eksternal.

        Args:
        X : List of List (m x l) - Input matrix (m samples, l features)
        W : List of List (l x n) - Weight matrix (l features, n output neurons)
        B : List (n) - Bias vector
        debug: Boolean - Debug mode

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
            print(f"\n=== Mulai iterasi i={i}, j={j} ===") if debug else None
            Z[i][j] = B[j]  # Pertama, masukkan nilai bias dulu ∑x*w + b == b + ∑x*w
            
            if debug:
                print(f"Z[{i}][{j}] start (bias): {Z[i][j]}")
            
            for k in range(l):  # Loop ketiga: iterasi tiap inputs untuk dot product
                print(f"  - Iterasi k={k}") if debug else None
                Z[i][j] += X[i][k] * W[k][j]
                if debug:
                    print(f"    + X[{i}][{k}] * W[{k}][{j}] => ({X[i][k]}) * ({W[k][j]}) = {X[i][k] * W[k][j]}")
                    print(f"      -> Z[{i}][{j}] now: {Z[i][j]}")
            
            # Cetak matriks Z yang ter-update
            if debug:
                print("\nMatriks Z setelah update:")
                for row in Z:
                    print(row)
                print("-" * 30)

    return Z