import numpy as np
import matplotlib.pyplot as plt

from math import pi

INF = 2
function = lambda x: np.where(abs(x) <= INF, x ** 2, 0)

# Define the interval and function and generate appropriate x values and y values
x_values = np.linspace(-10, 10, 1000)
y_values = function(x_values)

# Plot the original function
plt.figure(figsize=(12, 4))
plt.plot(x_values, y_values, label="Original y = x^2")
plt.title("Original Function (y = x^2)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig("Output1")
plt.show()


# Define the sampled times and frequencies
sampled_times = x_values
frequencies = x_values / 10

# Fourier Transform 
def fourier_transform(signal, frequencies, sampled_times):
    num_freqs = len(frequencies)
    ft_result_real = np.zeros(num_freqs)
    ft_result_imag = np.zeros(num_freqs)
    
    # Store the fourier transform results for each frequency. Handle the real and imaginary parts separately
    # use trapezoidal integration to calculate the real and imaginary parts of the FT

    def integration(func):
        return np.array([
            np.trapezoid(
                y_values * func(2 * pi * frequency * sampled_times),
                frequencies
            ) for frequency in frequencies
        ])

    ft_result_real = integration(np.cos)
    ft_result_imag = -integration(np.sin)

    return ft_result_real, ft_result_imag

# Apply FT to the sampled data
ft_data = fourier_transform(y_values, frequencies, sampled_times)
#  plot the FT data
plt.figure(figsize=(12, 6))
plt.plot(frequencies, np.sqrt(ft_data[0]**2 + ft_data[1]**2))
plt.title("Frequency Spectrum of y = x^2")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.savefig("Output2")
plt.show()


# Inverse Fourier Transform 
def inverse_fourier_transform(ft_signal, frequencies, sampled_times):
    n = len(sampled_times)
    reconstructed_signal = np.zeros(n)
    # Reconstruct the signal by summing over all frequencies for each time in sampled_times.
    # use trapezoidal integration to calculate the real part
    # You have to return only the real part of the reconstructed signal
    
    return reconstructed_signal

# Reconstruct the signal from the FT data
reconstructed_y_values = inverse_fourier_transform(ft_data, frequencies, sampled_times)
# Plot the original and reconstructed functions for comparison
plt.figure(figsize=(12, 4))
plt.plot(x_values, y_values, label="Original y = x^2", color="blue")
plt.plot(sampled_times, reconstructed_y_values, label="Reconstructed y = x^2", color="red", linestyle="--")
plt.title("Original vs Reconstructed Function (y = x^2)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig("Output3")
plt.show()
