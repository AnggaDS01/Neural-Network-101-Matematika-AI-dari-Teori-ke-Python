import time

from sum_arithmetic_formula import sum_arithmetic_formula
from sum_arithmetic_loop import sum_arithmetic_loop

def test_performance(start, end, method="both"):
    """
    Menguji kecepatan perhitungan dengan mencatat waktu eksekusi
    """
    print(f"\nMenghitung jumlah deret dari {start} sampai {end}:")
    
    if method in ["formula", "both"]:
        # Menggunakan rumus
        start_time = time.time()
        result = sum_arithmetic_formula(start, end)
        end_time = time.time()
        formula_time = (end_time - start_time) * 1000  # Konversi ke millisekon
        print(f"Menggunakan rumus:")
        print(f"Hasil: {result}")
        print(f"Waktu: {formula_time:.4f} ms")
    
    if method in ["loop", "both"]:
        # Menggunakan loop
        start_time = time.time()
        result = sum_arithmetic_loop(start, end)
        end_time = time.time()
        loop_time = (end_time - start_time) * 1000  # Konversi ke millisekon
        print(f"\nMenggunakan loop:")
        print(f"Hasil: {result}")
        print(f"Waktu: {loop_time:.4f} ms")