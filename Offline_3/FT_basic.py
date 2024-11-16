import numpy as np
import matplotlib.pyplot as plt

from math import pi

def integration(signal, func, var_domain, const_domain):
    return np.array([
        np.trapezoid(
            signal * func(2 * pi * point * const_domain),
            var_domain
        ) for point in var_domain
    ])

TIME_INF = 2
FREQUENCY_INF = 5
ACCURACY = 1000

PARABOLA = lambda x: x ** 2
TRIANGULAR = lambda x: 0.5 * (TIME_INF - abs(x))
SAWTOOTH = lambda x: TIME_INF + x
RECTANGULAR = lambda x: 1

current_function = SAWTOOTH
FUNCTION = lambda x: np.where(abs(x) <= TIME_INF, current_function(x), 0)

# Define the interval and function and generate appropriate x values and y values
x_values = np.linspace(-10, 10, ACCURACY)
y_values = FUNCTION(x_values)

# Plot the original function
plt.figure(figsize=(12, 4))
plt.plot(x_values, y_values, label="Original y = x^2")
plt.title("Original Function (y = x^2)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
# plt.savefig("Output1")
plt.show()


# Define the sampled times and frequencies
sampled_times = x_values
frequencies = np.linspace(-FREQUENCY_INF, FREQUENCY_INF, ACCURACY)

# Fourier Transform 
def fourier_transform(signal, frequencies, sampled_times):
    num_freqs = len(frequencies)
    ft_result_real = np.zeros(num_freqs)
    ft_result_imag = np.zeros(num_freqs)
    
    # Store the fourier transform results for each frequency. Handle the real and imaginary parts separately
    # use trapezoidal integration to calculate the real and imaginary parts of the FT

    ft_result_real += integration(signal, np.cos, frequencies, sampled_times)
    ft_result_imag += -integration(signal, np.sin, frequencies, sampled_times)

    return ft_result_real, ft_result_imag

# Apply FT to the sampled data
ft_data = fourier_transform(y_values, frequencies, sampled_times)
#  plot the FT data
plt.figure(figsize=(12, 6))
plt.plot(frequencies, np.sqrt(ft_data[0]**2 + ft_data[1]**2))
plt.title("Frequency Spectrum of y = x^2")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
# plt.savefig("Output2")
plt.show()


# Inverse Fourier Transform 
def inverse_fourier_transform(ft_signal, frequencies, sampled_times):
    n = len(sampled_times)
    reconstructed_signal = np.zeros(n)
    # Reconstruct the signal by summing over all frequencies for each time in sampled_times.
    # use trapezoidal integration to calculate the real part
    # You have to return only the real part of the reconstructed signal

    reconstructed_signal += integration(ft_signal[0], np.cos, sampled_times, frequencies)
    
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
# plt.savefig("Output3")
plt.show()
