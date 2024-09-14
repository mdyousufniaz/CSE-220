import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

INF = 8

def plot(
        signal, 
        title=None, 
        y_range=(-1, 3), 
        figsize = (8, 4), # original (8, 3) increased height to see x label
        x_label='n (Time Index)',
        y_label='x[n]',
        saveTo=None
    ):
    plt.figure(figsize=figsize)
    plt.xticks(np.arange(-INF, INF + 1, 1))
    
    y_range = (y_range[0], max(np.max(signal), y_range[1]) + 1)
    # set y range of 
    plt.ylim(*y_range)
    plt.stem(np.arange(-INF, INF + 1, 1), signal)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    if saveTo is not None:
        plt.savefig(saveTo)
    # plt.show()

def init_signal():
    return np.zeros(2 * INF + 1)


def time_reverse_signal(x : np.ndarray) -> np.ndarray:
    # implement this function
    return np.flip(x)

def odd_even_decomposition(x : np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    # implement this function
    # you can return 2 values from function in the following way
    # return value1, value2
    return 0.5 * (x - time_reverse_signal(x)), 0.5 * (x + time_reverse_signal(x))

def main():
    img_root_path = '.'
    signal = init_signal()
    signal[INF] = 1
    signal[INF+1] = .5
    signal[INF-1] = 2
    signal[INF + 2] = 1
    signal[INF - 2] = .5

    plot(signal, title='Original Signal(x[n])', saveTo=f'{img_root_path}/x[n].png')
    reversed = time_reverse_signal(signal)
    plot(reversed, title='x[-n]', saveTo=f'{img_root_path}/x[-n].png')
    plot(time_reverse_signal(reversed), title='x[-(-n)]', saveTo=f'{img_root_path}/x[-(-n)].png')
    odd_signal, even_signal = odd_even_decomposition(signal)
    plot(odd_signal, title='Odd Signal(x[n])', saveTo=f'{img_root_path}/x_odd[n].png')
    plot(even_signal, title='Even Signal(x[n])', saveTo=f'{img_root_path}/x_even[n].png')


main()