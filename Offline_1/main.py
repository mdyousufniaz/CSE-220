from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from typing import Callable, Optional
from fractions import Fraction

import os

DIS_INF = 5
CONT_INF = 3
DELTA = 'Δ'
SIGMA = 'Σ'

def configure_plot(
        x_label: str,
        y_label: str,
        title: str = 'title',
        figsize: tuple[float, float] = (6, 4)
) -> None:
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)

def save_plot(save_to: str):
        plt.savefig(save_to)
        plt.close()


class DiscreteSignal:

    def __init__(self, INF: Fraction) -> None:
        self.INF = INF
        self.values = np.zeros(2 * self.INF + 1)

    def set_value_at_time(self, time: Fraction, value: Fraction) -> None:
        # Assuming the time will not go out of range

        self.values[self.INF + time] = value

    def shift_signal(self, shift: Fraction) -> DiscreteSignal:
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
    
    def multiply_const_factor(self, scaler: Fraction) -> DiscreteSignal:
        new_discrete_signal = DiscreteSignal(self.INF)

        new_discrete_signal.values = self.values * scaler
        return new_discrete_signal
    
    def plot(
            self,
            title: str = 'Title',
            x_label: str ="n (Time Index)",
            y_label: str = 'x[n]',
            save_to: Optional[str] = None
    ) -> None:
        x_indexes = np.arange(-self.INF, self.INF + 1)

        configure_plot(x_label, y_label, title)
        plt.xticks(x_indexes)
        plt.ylim(-1, max(np.max(self.values), 3) + 1)
        plt.stem(x_indexes, self.values)
        if save_to:
            save_plot(save_to)
        
        

class ContinuousSignal:

    def __init__(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        self.func = func

    def shift(self, shift: Fraction) -> ContinuousSignal:
        return ContinuousSignal(lambda t: self.func(t - shift))

    def add(self, other: ContinuousSignal) -> ContinuousSignal:
        return ContinuousSignal(lambda t: self.func(t) + other.func(t))
    
    def multiply(self, other: ContinuousSignal) -> ContinuousSignal:
        return ContinuousSignal(lambda t: self.func(t) * other.func(t))
    
    def multiply_const_factor(self, scaler: Fraction) -> ContinuousSignal:
        return ContinuousSignal(lambda t: self.func(t) * scaler)
    
    def plot_signal(self, label: Optional[str] = None) -> None:
        time_stamps = np.linspace(-CONT_INF, CONT_INF, 500)
        plt.ylim(0, max(1.2, self.func(CONT_INF) + 0.2))
        plt.plot(
            time_stamps,
            self.func(time_stamps),
            label=label
        )

    def plot(
            self,
            title: str = 'Title',
            x_label: str ="n (Time Index)",
            y_label: str = 'x[n]',
            save_to: Optional[str] = None
    ) -> None:
        configure_plot(x_label, y_label, title)
        self.plot_signal()

        if save_to:
            save_plot(save_to)
        
class LTI_Discreate:

    def __init__(self, impulse_response: DiscreteSignal) -> None:
        self.impulse_response = impulse_response
    
    def linear_combination_of_impulses(self, input_signal: DiscreteSignal):
        INF = input_signal.INF
        unit_impluse = DiscreteSignal(INF)
        unit_impluse.set_value_at_time(0, 1)

        return tuple(unit_impluse.shift_signal(time) for time in range(-INF, INF + 1)), input_signal.values
    
    def output(self, input_signal: DiscreteSignal):
        INF = input_signal.INF
        systems_responses = []
        output_signal = DiscreteSignal(INF)

        for time in range(-INF, INF + 1):
            systems_response = self.impulse_response.shift_signal(time)
            if (coffecient := input_signal.values[time + INF]):
                output_signal = output_signal.add(systems_response.multiply_const_factor(coffecient))

            systems_responses.append(systems_response)

        return output_signal, systems_responses, input_signal.values
        

class LTI_Continuous:

    def __init__(self, impulse_response: ContinuousSignal) -> None:
        self.impulse_response = impulse_response

    def linear_combination_of_impulses(self, input_signal: ContinuousSignal, delta: Fraction):
        unit_impulses = []
        coefficients = []
        unit_impulse = ContinuousSignal(lambda t: np.where((t >= 0) & (t <= delta), 1, 0))

        start = int(CONT_INF // delta)

        for time in range(-start, start):
            index = time * delta
            unit_impulses.append(unit_impulse.shift(index))
            coefficients.append(input_signal.func(index))

        return unit_impulses, coefficients
    
    def output_approx(self, input_signal: ContinuousSignal, delta: Fraction):
        _, coefficients = self.linear_combination_of_impulses(input_signal, delta)
        output_signal = ContinuousSignal(lambda t: np.zeros_like(t))
        systems_responses = []

        for time, coefficient in enumerate(coefficients, -int(CONT_INF // delta)):
            # print(coefficient, time)
            if coefficient:
                systems_response = self.impulse_response.shift(time * delta)
                # systems_response.plot()
                output_signal = output_signal.add(systems_response.multiply_const_factor(coefficient * delta))
            else:
                systems_response = ContinuousSignal(lambda t: np.zeros_like(t))

            systems_responses.append(systems_response)

        return output_signal, systems_responses, coefficients
    
    def compare_plot(
            self,
            org_signal: ContinuousSignal,
            org_label: str,
            recon_signal: ContinuousSignal,
            recon_label: str, 
            delta: Fraction,
            save_to: str
    ):
        configure_plot(
            x_label="t (Time)",
            y_label="x(t)",
            title=f"{DELTA} = {delta}"
        )
        recon_signal.plot_signal(recon_label)
        org_signal.plot_signal(org_label)
        plt.legend()
        
        save_plot(save_to)


def main():
    # Create image folders for discrete and continuous plots
    discrete_folder = "discrete_plots"
    continuous_folder = "continuous_plots"

    if not os.path.exists(discrete_folder):
        os.makedirs(discrete_folder)

    if not os.path.exists(continuous_folder):
        os.makedirs(continuous_folder)

    # 6.1 Discrete Portion
    dis_signal_1 = DiscreteSignal(DIS_INF)
    for i in range(3):
        dis_signal_1.set_value_at_time(i, 1)

    dis_signal_2 = DiscreteSignal(DIS_INF)
    dis_signal_2.set_value_at_time(0, 0.5)
    dis_signal_2.set_value_at_time(1, 2)

    lti_discrete = LTI_Discreate(dis_signal_1)
    index = -DIS_INF
    img_index = 0

    linear_combination_folder = os.path.join(discrete_folder, 'linear_combination')
    os.makedirs(linear_combination_folder, exist_ok=True)

    for signal, coff in zip (*lti_discrete.linear_combination_of_impulses(dis_signal_2)):
        signal.multiply_const_factor(coff).plot(
            title=f'{SIGMA}[n - ({index})]x[{index}]',
            save_to=f'{linear_combination_folder}/image_{img_index}'
        )
        index += 1
        img_index += 1

    output, signals, coffs = lti_discrete.output(dis_signal_2)
    output.plot(
        'output',
        save_to=f"{discrete_folder}/output"
    )

    systems_response_folder = os.path.join(discrete_folder, 'systems_response')
    os.makedirs(systems_response_folder, exist_ok=True)


    for signal, coff in zip(signals, coffs):
        signal.multiply_const_factor(coff).plot(
            title=f'h[n - ({index})] * x[{index}]',
            save_to=f'{systems_response_folder}/image_{img_index}'
        )
        index += 1
        img_index += 1
    
    # continous portion
    cont_signal_1 = ContinuousSignal(lambda t: np.where(t >= 0, 1, 0))
    cont_signal_2 = ContinuousSignal(lambda t: np.where(t >= 0, np.exp(-t), 0))
    cont_signal_3 = ContinuousSignal(lambda t: np.where(t >= 0, 1 - np.exp(-t), 0))

    lti_Continuous = LTI_Continuous(cont_signal_1)

    sum = ContinuousSignal(lambda t: np.zeros_like(t))
    deltas = 0.5, 0.1, 0.05, 0.01 

    linear_combination_folder = os.path.join(continuous_folder, 'linear_combination')
    os.makedirs(linear_combination_folder, exist_ok=True)

    index = int(-CONT_INF // deltas[0])
    img_index = 0

    for signal, coff in zip (*lti_Continuous.linear_combination_of_impulses(cont_signal_2, deltas[0])):
        temp_signal = signal.multiply_const_factor(coff)
        sum = sum.add(temp_signal)
        temp_signal.plot(
            title=f'δ[t - ({index}{DELTA})]x({index}{DELTA}){DELTA}',
            save_to=f'{linear_combination_folder}/image_{img_index}'
        )
        index += 1
        img_index += 1

    sum.plot(
            title=f'Reconstructed Signal',
            save_to=f'{linear_combination_folder}/Reconstructed_Signal'
        )
    
    folder_1 = os.path.join(continuous_folder, 'folder_1')
    os.makedirs(folder_1, exist_ok=True)
    
    lti_Continuous.compare_plot(cont_signal_2, 'x(t)', sum, 'Reconstructed', deltas[0], f'{folder_1}/img_{1}')
    
    for i, delta in enumerate(deltas[1:], start=2):  # Start index at 2 for naming
        
        sum = ContinuousSignal(lambda t: np.zeros_like(t))
        index = int(-CONT_INF // delta)
        img_index = 0

        for signal, coff in zip (*lti_Continuous.linear_combination_of_impulses(cont_signal_2, delta)):
            sum = sum.add(signal.multiply_const_factor(coff))
            index += 1
            img_index += 1

        lti_Continuous.compare_plot(
            cont_signal_2,
            'x(t)',
            sum,
            'Reconstructed',
            delta,
            f'{folder_1}/img_{i}.png'  # Save in folder_1 with proper index and extension
        )

    output, signals, coffs = lti_Continuous.output_approx(cont_signal_2, deltas[0])

    systems_response_folder = os.path.join(continuous_folder, 'systems_response')
    os.makedirs(systems_response_folder, exist_ok=True)

    index = int(-CONT_INF // deltas[0])
    img_index = 0

    for signal, coff in zip(signals, coffs):
        signal.multiply_const_factor(coff * deltas[0]).plot(
            title=f'h[t - ({index}{DELTA})] * x({index}{DELTA}){DELTA}',
            save_to=f'{systems_response_folder}/image_{img_index}'
        )

        index += 1
        img_index += 1


    folder_2 = os.path.join(continuous_folder, 'folder_2')
    os.makedirs(folder_2, exist_ok=True)

    output.plot(
        'output_1',
        save_to=f"{continuous_folder}/output"
    ) 

    for i, delta in enumerate(deltas, start=1):
        output, signals, coffs = lti_Continuous.output_approx(cont_signal_2, delta)

        lti_Continuous.compare_plot(
            cont_signal_3,
            'y(t)= (1 - e ^(-t))u(t)',
            output,
            'y_approx(t)',
            delta,
            f'{folder_2}/output_{i}'
        )



if __name__ == "__main__":
    main()
    

        
        