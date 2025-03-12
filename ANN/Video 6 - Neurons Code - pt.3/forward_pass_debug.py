def forward_pass_debug(
        X: list, 
        W: list, 
        B: list, 
        debug: bool = False
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
        debug (bool, optional): Mode debug untuk menampilkan proses perhitungan. 
                               Default adalah False.

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
            if debug:
                print(f"\n=== Mulai iterasi i={i}, j={j} ===")
            
            # Inisialisasi Z[i][j] dengan bias B[j]
            # Kenapa bias ditambahkan dulu? Karena rumus transformasi linear adalah Z = X * W + B.
            Z[i][j] = B[j]  
            
            if debug:
                print(f"Z[{i}][{j}] start (bias): {Z[i][j]}")
            
            for k in range(l):  # Loop ketiga: iterasi tiap fitur (kolom di X dan baris di W)
                if debug:
                    print(f"  - Iterasi k={k}")
                
                # Hitung dot product: X[i][k] * W[k][j]
                Z[i][j] += X[i][k] * W[k][j]
                
                if debug:
                    print(f"    + X[{i}][{k}] * W[{k}][{j}] => ({X[i][k]}) * ({W[k][j]}) = {X[i][k] * W[k][j]}")
                    print(f"      -> Z[{i}][{j}] now: {Z[i][j]}")
            
            # Cetak matriks Z yang ter-update setelah setiap iterasi
            if debug:
                print("\nMatriks Z setelah update:")
                for row in Z:
                    print(row)
                print("-" * 30)

    return Z