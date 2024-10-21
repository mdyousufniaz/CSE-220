from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

class DiscreteSignal:

    def __init__(self, INF: int) -> None:
        self.INF = INF
        self.values = np.zeros(2 * self.INF + 1)

    def set_value_at_time(self, time: int, value: int) -> None:
        # Assuming the time will not go out of range

        self.values[self.INF + time] = value

    def shift_signal(self, shift: int) -> DiscreteSignal:
        new_discrete_signal = DiscreteSignal(self.INF)

        if shift > 0:
            new_discrete_signal.values = np.concatenate((np.zeros(shift), self.values[:-shift]))
        elif shift < 0:
            shift = abs(shift)
            new_discrete_signal.values = np.concatenate((self.values[shift:], np.zeros(shift)))
        else:
            new_discrete_signal.values = self.values
        
        return new_discrete_signal

    def add(self, other: DiscreteSignal) -> DiscreteSignal:
        new_discrete_signal = DiscreteSignal(self.INF)

        new_discrete_signal.values = self.values + other.values
        return new_discrete_signal
    
    def multiply(self, other: DiscreteSignal) -> DiscreteSignal:
        new_discrete_signal = DiscreteSignal(self.INF)

        new_discrete_signal.values = self.values * other.values
        return new_discrete_signal
    
    def multiply_const_factor(self, scaler: int) -> DiscreteSignal:
        new_discrete_signal = DiscreteSignal(self.INF)

        new_discrete_signal.values = self.values * scaler
        return new_discrete_signal
    
    def plot(
            self,
            title: str,
            x_label: str,
            y_label: str
    ) -> None:
        x_indexes = np.arange(-self.INF, self.INF + 1)

        plt.figure(figsize=(8, 4))
        plt.xticks(x_indexes)
        plt.ylim(-1, max(np.max(self.values), 3) + 1)
        plt.stem(x_indexes, self.values)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)

class ContinuousSignal:

    def __init__(self, func: function) -> None:
        self.func = func

    def shift(self, shift) -> ContinuousSignal:
        pass