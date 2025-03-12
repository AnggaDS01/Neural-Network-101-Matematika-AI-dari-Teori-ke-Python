def sum_arithmetic_loop(start, end):
    """
    Menghitung jumlah deret aritmetika menggunakan perulangan.

    Kenapa pake loop? Untuk membandingkan efisiensi dengan rumus.
    Loop punya kompleksitas O(n), jadi lebih lambat untuk range besar.

    Args:
        start (int): Angka awal deret.
        end (int): Angka akhir deret.

    Returns:
        int: Jumlah total deret.
    """
    total = 0  # Inisialisasi variabel untuk menyimpan total
    for i in range(start, end + 1):  # Loop dari start sampai end
        total += i  # Tambahkan setiap elemen ke total
    return total  # Kembalikan hasil penjumlahan