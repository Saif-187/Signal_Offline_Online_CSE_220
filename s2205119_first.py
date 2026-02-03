import numpy as np
import matplotlib.pyplot as plt

class Signal:
    def __init__(self, INF):
        self.INF = INF
        self.n = np.arange(-INF, INF + 1)
        self.values = np.zeros(len(self.n))

    def set_value_at_time(self, t, value):
        if -self.INF <= t <= self.INF:
            self.values[t + self.INF] = value

    def shift(self, k):
        # x(n-k)
        shifted = Signal(self.INF)
        for i, n in enumerate(self.n):
            if -self.INF <= n - k <= self.INF:
                shifted.values[i] = self.values[(n - k) + self.INF]
        return shifted

    def add(self, other):
        result = Signal(self.INF)
        result.values = self.values + other.values
        return result

    def multiply(self, scalar):
        result = Signal(self.INF)
        result.values = scalar * self.values
        return result

    def plot(self, title="Discrete Signal"):
        plt.stem(self.n, self.values)
        plt.xlabel("n")
        plt.ylabel("Amplitude")
        plt.title(title)
        plt.grid(True)
        plt.show()


class LTI_System:
    def __init__(self, impulse_response: Signal):
        self.h = impulse_response

    def linear_combination_of_impulses(self, input_signal: Signal):
        impulses = []
        coefficients = []

        for i, n in enumerate(input_signal.n):
            value = input_signal.values[i]
            if value != 0:
                delta = Signal(input_signal.INF)
                delta.set_value_at_time(n, 1)
                impulses.append(delta)
                coefficients.append(value)

        return impulses, coefficients

    def output(self, input_signal: Signal):
        impulses, coefficients = self.linear_combination_of_impulses(input_signal)
        y = Signal(input_signal.INF)

        for delta, coeff in zip(impulses, coefficients):
            k = delta.n[delta.values == 1][0]
            shifted_h = self.h.shift(k)
            y = y.add(shifted_h.multiply(coeff))

        return y


if __name__ == "__main__":
    INF = 10

    # Input signal x(n)
    x = Signal(INF)
    x.set_value_at_time(-2, 1)
    x.set_value_at_time(0, 2)
    x.set_value_at_time(3, -1)
    x.plot("Input Signal x(n)")

    # Impulse response h(n)
    h = Signal(INF)
    h.set_value_at_time(0, 1)
    h.set_value_at_time(1, 0.5)
    #h.plot("Impulse Response h(n)")

    # LTI system
    system = LTI_System(h)

    # Output
    y = system.output(x)
    y.plot("Output Signal y(n)")
