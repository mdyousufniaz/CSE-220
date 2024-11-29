import numpy as np
import matplotlib.pyplot as plt

from math import pi
from timeit import timeit

from util import DFT, IDFT

def random_dts_generator(k: int) -> np.ndarray:
    return 2 * np.random.rand(2**k) - 1

def FT_helper(dts: np.ndarray, coff: int) -> np.ndarray:
    if dts.size == 1: return dts
    
    even: np.ndarray = dts[::2]
    odd: np.ndarray = dts[1::2]

    even_fft = FT_helper(even, coff)
    odd_fft = FT_helper(odd, coff)

    N = dts.size
    result: np.ndarray = np.empty(N, dtype=complex)

    for k in range(N // 2):
        twiddle_factor = np.exp(coff * 1j * 2 * pi * k / N)
        result[k] = even_fft[k] + twiddle_factor * odd_fft[k]
        result[k + N // 2] = even_fft[k] - twiddle_factor * odd_fft[k]

    return result

def FFT(dst: np.ndarray) -> np.ndarray:
    return FT_helper(dst, -1)

def IFFT(dst: np.ndarray) -> np.ndarray:
    return FT_helper(dst, 1) * 1 / dst.size

def plot_comparison_graph(
        x_values,
        runtimes_1,
        runtimes_2,
        transfrom_type_1,
        transfrom_type_2
):
    plt.plot(x_values, runtimes_1, color="blue", label=transfrom_type_1)
    plt.plot(x_values, runtimes_2, color="red", label=transfrom_type_2)
    plt.xlabel("Input size(n)")
    plt.ylabel("Run Times(s)")
    plt.title(f"{transfrom_type_1} vs {transfrom_type_2} Runtime Analysis")
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_avg_runtime(transfrom_func, signal, exec_num=10):
    return timeit(lambda: transfrom_func(signal), number=exec_num) / exec_num

def perform_analysis(func_1, func_2, inverse):
    START = 2
    END = 10
    DIFF = END - START + 1
    POWER_RANGE = range(START, END + 1)

    runtimes_1 = np.empty(DIFF)
    runtimes_2 = np.empty(DIFF)

    X_VALUES = np.array(tuple(2**k for k in POWER_RANGE))

    for index, k in enumerate(POWER_RANGE):
        input_signal = random_dts_generator(k)
        if inverse: input_signal = IFFT(input_signal)

        runtimes_1[index] = compute_avg_runtime(func_1, input_signal)
        runtimes_2[index] = compute_avg_runtime(func_2, input_signal)

    # print(n_values)
    plot_comparison_graph(
        X_VALUES,
        runtimes_1,
        runtimes_2,
        ('I' if inverse else '') + "DFT",
        ('I' if inverse else '') + "FFT"
    )

perform_analysis(DFT, FFT, False)
perform_analysis(IDFT, IFFT, True)

