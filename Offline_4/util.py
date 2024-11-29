import numpy as np
import matplotlib.pyplot as plt

from math import pi

def FT(signal, N, coff):
    return np.array([
        sum(signal * np.exp(coff* 1j * 2 * pi * np.arange(N) * c / N)) for c in range(N)
    ])

def DFT(signal):
    return FT(signal, signal.size, -1)

def IDFT(signal):
    N = signal.size
    return FT(signal, N, 1) / N

def plot_signal(
        x_values,
        y_values,
        title,
        y_label,
        color='green',
        x_label="Sample Index",
        plot_width=6
):
    plt.figure(figsize=(plot_width, 6))
    plt.stem(x_values, y_values, color)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_DTS(
        x_values,
        y_values,
        name,
        color
):
    return plot_signal(
        x_values,
        y_values,
        f"Signal {name} (Station {name})",
        "Amplitude",
        color
    )

def plot_MS(
        x_values,
        signal,
        name,
        color
):
    return plot_signal(
        x_values,
        np.sqrt(signal.real**2 + signal.imag**2),
        f"Magnitude Spectrum of Signal {name}",
        "Magnitude",
        color,
        plot_width=8
    )

def plot_cross_correlation(
        x_values,
        y_values
):
    return plot_signal(
        x_values,
        y_values,
        "DFT-based Cross-Correlation",
        "Correlation",
        x_label="Lag(samples)"
    )
