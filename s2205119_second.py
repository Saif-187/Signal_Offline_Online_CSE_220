import matplotlib.pyplot as plt
from s2205119_first import Signal, LTI_System


def read_signal_from_file(filename, INF):
    with open(filename, "r") as f:
        nstart, nend = map(int, f.readline().split())
        values = list(map(float, f.readline().split()))

    signal = Signal(INF)
    for i, n in enumerate(range(nstart, nend + 1)):
        signal.set_value_at_time(n, values[i])

    return signal


if __name__ == "__main__":
    INF = 50

    # Read noisy signal
    x = read_signal_from_file("input_signal.txt", INF)
    x.plot("Noisy Input Signal")

    # 5-point moving average impulse response
    h = Signal(INF)
    for n in range(-2, 3):
        h.set_value_at_time(n, 1 / 5)

    #h.plot("Impulse Response (Moving Average Filter)")

    # LTI system
    system = LTI_System(h)

    # Smoothed output
    y = system.output(x)
    y.plot("Smoothed Output Signal")
