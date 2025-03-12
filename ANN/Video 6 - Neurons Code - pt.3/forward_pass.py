def forward_pass(
    X: list, 
    W: list, 
    B: list, 
) -> list:
    """
    Melakukan forward pass sederhana tanpa library eksternal.

    Kenapa pake implementasi manual? Untuk memahami konsep dasar 
    perkalian matriks dan transformasi linear dalam neural network.

    Args:
        X (list of list): Matriks input berukuran (m x l), di mana:
                           - m = jumlah sampel
                           - l = jumlah fitur
        W (list of list): Matriks weight berukuran (l x n), di mana:
                           - l = jumlah fitur (harus sama dengan X)
                           - n = jumlah neuron di layer berikutnya
        B (list): Vektor bias berukuran (n), di mana:
                  - n = jumlah neuron di layer berikutnya

    Returns:
        Z (list of list): Matriks output berukuran (m x n), di mana:
                          - m = jumlah sampel
                          - n = jumlah neuron di layer berikutnya
    """
    m = len(X)     # Jumlah sampel (baris di X)
    l = len(X[0])  # Jumlah fitur (kolom di X)
    n = len(B)     # Jumlah neuron di layer berikutnya (kolom di W)

    # Inisialisasi matriks output Z dengan ukuran (m x n)
    # Kenapa diinisialisasi dengan 0? Karena kita akan mengakumulasi hasil perkalian.
    Z = [[0] * n for _ in range(m)]

    # Perkalian matriks manual: Z = X * W + B
    for i in range(m):      # Loop pertama: iterasi tiap sampel (baris di X)
        for j in range(n):  # Loop kedua: iterasi tiap neuron (kolom di W)
            Z[i][j] = B[j]  # Inisialisasi Z[i][j] dengan bias B[j]
            for k in range(l):  # Loop ketiga: iterasi tiap fitur (kolom di X dan baris di W)
                # Hitung dot product: X[i][k] * W[k][j]
                Z[i][j] += X[i][k] * W[k][j]

    return Z