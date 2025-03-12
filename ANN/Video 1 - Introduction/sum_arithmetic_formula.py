def sum_arithmetic_formula(start, end):
    """
    Menghitung jumlah deret aritmetika menggunakan rumus matematika.

    Kenapa pake rumus? Karena lebih efisien secara komputasi, 
    terutama untuk deret dengan range besar. Kompleksitasnya O(1).

    Rumus yang digunakan: 
    Jumlah deret = n * (a + b) / 2
    di mana:
    - n = jumlah elemen deret
    - a = elemen pertama
    - b = elemen terakhir

    Args:
        start (int): Angka awal deret.
        end (int): Angka akhir deret.

    Returns:
        int: Jumlah total deret.
    """
    n = end - start + 1  # Hitung jumlah elemen deret
    return (n * (start + end)) // 2  # Gunakan rumus untuk menghitung jumlah deret