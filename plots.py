import matplotlib.pyplot as plt

plt.plot(df["dim"], df["speedup_numba"], label="Numba")
plt.plot(df["dim"], df["speedup_cython"], label="Cython")
plt.xlabel("Dimension (seq_len)")
plt.ylabel("Speed-up vs NumPy")
plt.legend()
plt.title("Comparaison des accélérations")
plt.show()
