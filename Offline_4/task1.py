import numpy as np

from util import DFT, IDFT, plot_DTS, plot_MS, plot_cross_correlation
n=50
samples = np.arange(n) 
sampling_rate=100
wave_velocity=8000



#use this function to generate signal_A and signal_B with a random shift
def generate_signals(frequency=5):

    noise_freqs = [15, 30, 45]  # Default noise frequencies in Hz

    amplitudes = [0.5, 0.3, 0.1]  # Default noise amplitudes
    noise_freqs2 = [10, 20, 40] 
    amplitudes2 = [0.3, 0.2, 0.1]
    
     # Discrete sample indices
    dt = 1 / sampling_rate  # Sampling interval in seconds
    time = samples * dt  # Time points corresponding to each sample

    # Original clean signal (sinusoidal)
    original_signal = np.sin(2 * np.pi * frequency * time)

    # Adding noise
    noise_for_sigal_A = sum(amplitude * np.sin(2 * np.pi * noise_freq * time)
                for noise_freq, amplitude in zip(noise_freqs, amplitudes))
    noise_for_sigal_B = sum(amplitude * np.sin(2 * np.pi * noise_freq * time)
                for noise_freq, amplitude in zip(noise_freqs2, amplitudes2))
    signal_A = original_signal + noise_for_sigal_A 
    noisy_signal_B = signal_A + noise_for_sigal_B

    # Applying random shift
    shift_samples = np.random.randint(-n // 2, n // 2)  # Random shift
    # shift_samples = 24
    print(f"Shift Samples: {shift_samples}")
    signal_B = np.roll(noisy_signal_B, shift_samples)
    
    return signal_A, signal_B

#implement other functions and logic

signal_A, signal_B = generate_signals()

plot_DTS(samples, signal_A, 'A', 'blue')

dft_signal_A = DFT(signal_A)
plot_MS(samples, dft_signal_A, 'A', 'blue')

plot_DTS(samples, signal_B, 'B', 'red')

dft_signal_B = DFT(signal_B)
plot_MS(samples, dft_signal_B, 'B', 'red')

bias = n // 2 - 1
r_n = np.roll(IDFT(dft_signal_A * np.conj(dft_signal_B)).real, bias)

plot_cross_correlation(bias - samples, r_n)

sample_lag = int(bias - np.argmax(r_n))
print(f"{sample_lag = }")
distance = float(abs(sample_lag) * wave_velocity / sampling_rate)
print(f"{distance = }")

plot_DTS(samples, np.roll(signal_B, -sample_lag), 'Aligned', 'midnightblue')


