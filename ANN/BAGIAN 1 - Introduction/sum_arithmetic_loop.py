def sum_arithmetic_loop(start, end):
    """
    Menghitung jumlah deret aritmetika menggunakan loop
    """
    total = 0
    for i in range(start, end + 1):
        total += i
    return total