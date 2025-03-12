import time

from sum_arithmetic_formula import sum_arithmetic_formula
from sum_arithmetic_loop import sum_arithmetic_loop

def test_performance(start, end, method="both"):
    """
    Menguji performa perhitungan jumlah deret aritmatika dengan dua metode:
    1. Menggunakan rumus matematika (sum_arithmetic_formula)
    2. Menggunakan perulangan (sum_arithmetic_loop)

    Kenapa perlu diuji? Karena kita ingin tahu metode mana yang lebih efisien 
    dalam hal waktu eksekusi, terutama untuk deret dengan range yang besar.

    Args:
        start (int): Angka awal deret.
        end (int): Angka akhir deret.
        method (str, optional): Metode yang ingin diuji. Pilihan: "formula", "loop", atau "both". 
                                Default adalah "both" (uji kedua metode).

    Returns:
        None: Fungsi ini hanya mencetak hasil dan waktu eksekusi.
    """
    print(f"\nMenghitung jumlah deret dari {start} sampai {end}:")
    
    if method in ["formula", "both"]:
        # Menggunakan rumus matematika untuk menghitung jumlah deret.
        # Kenapa pake rumus? Karena lebih efisien secara komputasi, 
        # terutama untuk deret dengan range besar. Kompleksitasnya O(1).
        start_time = time.time()  # Catat waktu mulai
        result = sum_arithmetic_formula(start, end)  # Panggil fungsi rumus
        end_time = time.time()  # Catat waktu selesai
        formula_time = (end_time - start_time) * 1000  # Konversi ke millisekon
        print(f"Menggunakan rumus:")
        print(f"Hasil: {result}")
        print(f"Waktu: {formula_time:.4f} ms")  # Cetak waktu eksekusi
    
    if method in ["loop", "both"]:
        # Menggunakan perulangan untuk menghitung jumlah deret.
        # Kenapa pake loop? Untuk membandingkan efisiensi dengan rumus.
        # Loop punya kompleksitas O(n), jadi lebih lambat untuk range besar.
        start_time = time.time()  # Catat waktu mulai
        result = sum_arithmetic_loop(start, end)  # Panggil fungsi loop
        end_time = time.time()  # Catat waktu selesai
        loop_time = (end_time - start_time) * 1000  # Konversi ke millisekon
        print(f"\nMenggunakan loop:")
        print(f"Hasil: {result}")
        print(f"Waktu: {loop_time:.4f} ms")  # Cetak waktu eksekusi