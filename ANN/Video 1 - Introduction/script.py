from sum_arithmetic_formula import sum_arithmetic_formula
from sum_arithmetic_loop import sum_arithmetic_loop
from test_performance import test_performance

print("=== Perbandingan Kecepatan Penjumlahan Deret Aritmetika ===")

# Uji performa untuk range kecil (1 sampai 10)
test_performance(1, 10)
# Kenapa diuji range kecil? Untuk memastikan kedua metode memberikan hasil yang sama.

# Uji performa untuk range besar (1 sampai 1.000.000)
test_performance(1, 1_000_000)
# Kenapa diuji range besar? Untuk melihat perbedaan performa yang signifikan 
# antara metode rumus dan loop.