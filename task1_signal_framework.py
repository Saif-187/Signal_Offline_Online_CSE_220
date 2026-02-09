import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Abstract Base Class for Continuous-Time Signals
# =====================================================
class ContinuousSignal:
    """
    Abstract base class for all continuous-time signals.
    Every signal must be defined over a time axis t.
    """

    def __init__(self, t):
        self.t = t

    def values(self):
        """
        Returns the signal values evaluated over time axis t.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def plot(self, title="Signal"):
        """
        Plot the signal in the time domain.
        """
        plt.plot(self.t, self.values())
        plt.xlabel("Time (t)")
        plt.ylabel("Amplitude")
        plt.title(title)
        plt.grid(True)
        plt.show()


# =====================================================
# Signal Generator Class
# =====================================================
class SignalGenerator(ContinuousSignal):
    """
    Generates various continuous-time signals.
    Each method returns a numpy array of signal samples.
    """

    def sine(self, amplitude, frequency):
        """Generate a sine wave."""
        return np.sin(2 * np.pi * frequency * self.t) * amplitude

    def cosine(self, amplitude, frequency):
        """Generate a cosine wave."""
        return np.cos(2 * np.pi * frequency * self.t) * amplitude

    def square(self, amplitude, frequency):
        """Generate a square wave using sign of sine."""
        return amplitude * np.sign(np.sin(2 * np.pi * frequency * self.t))

    def sawtooth(self, amplitude, frequency):
        """Generate a sawtooth wave."""
        freq = frequency * self.t
        return amplitude * (2 * (freq - np.floor(0.5 + freq)))

    def triangle(self, amplitude, frequency):
        """Generate a triangle wave."""
        saw=self.sawtooth(1, frequency)
        return amplitude * (2 * np.abs(saw) - 1)
    def cubic(self, coefficient):
        """Generate a cubic polynomial signal."""
        return coefficient * self.t**3

    def parabolic(self, coefficient):
        """Generate a parabolic signal."""
        return coefficient * self.t**2

    def rectangular(self, width):
        """Generate a rectangular window centered at t=0."""
        st=np.abs(self.t)
        return np.where(st <= width / 2, 1, 0)

    def pulse(self, start, end):
        """Generate a finite pulse active between start and end."""
        flag=(self.t >= start) & (self.t <= end)
        return np.where(flag, 1, 0)


# =====================================================
# Composite Signal Class
# =====================================================
class CompositeSignal(ContinuousSignal):
    """
    Combines multiple signals into a single composite signal.
    """

    def __init__(self, t):
        super().__init__(t)
        self.components = []

    def add_component(self, signal):
        """
        Add a signal component to the composite signal.
        """
        self.components.append(signal)

    def values(self):
        """
        Sum all signal components.
        """
        return np.sum(self.components, axis=0)


# =====================================================
# Continuous Fourier Transform Analyzer
# =====================================================
class CFTAnalyzer:
    """
    Computes the Continuous Fourier Transform (CFT)
    using numerical integration (np.trapz).
    """

    def __init__(self, signal, t, frequencies):
        self.signal = signal
        self.t = t
        self.frequencies = frequencies

    def compute_cft(self):
        """
        Compute real and imaginary parts of the CFT.
        """
        x=self.signal.values()
        real=[]
        imag=[]
        for f in self.frequencies:
            cos=np.cos(2 * np.pi * f * self.t)
            sin=np.sin(2 * np.pi * f * self.t)
            real.append(np.trapezoid(x * cos, self.t))
            imag.append(-np.trapezoid(x * sin, self.t))
        return np.array(real),np.array(imag)

    def plot_spectrum(self):
        """
        Plot magnitude spectrum of the signal.
        """
        real , imag = self.compute_cft()
        magnitude = np.sqrt(real**2 + imag**2)
        plt.plot(self.frequencies, magnitude)
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.title("CFT Magnitude Spectrum")
        plt.grid()
        plt.show()


# =====================================================
# Inverse Continuous Fourier Transform
# =====================================================
class InverseCFT:
    """
    Reconstructs time-domain signal using ICFT.
    """

    def __init__(self, spectrum, frequencies, t):
        self.spectrum = spectrum
        self.frequencies = frequencies
        self.t = t

    def reconstruct(self):
        """
        Perform inverse CFT using numerical integration.
        """
        x=[]
        for i in self.t:
            cos_term = np.cos(2*np.pi*self.frequencies*i)
            sin_term = np.sin(2*np.pi*self.frequencies*i)
            x.append(np.trapezoid(self.spectrum[0]*cos_term - self.spectrum[1]*sin_term, self.frequencies))
        return np.array(x)
# =====================================================
# Main Execution (Task 1)
# =====================================================
t = np.linspace(-4, 4, 3000)
gen = SignalGenerator(t)

composite = CompositeSignal(t)
composite.add_component(gen.sine(2, 1))
composite.add_component(gen.cosine(0.5, 3))
composite.add_component(gen.square(1, 1))
composite.add_component(gen.cubic(1) * gen.rectangular(2))

composite.plot("Composite Signal")

frequencies = np.linspace(-10, 10, 1000)
cft = CFTAnalyzer(composite, t, frequencies)
cft.plot_spectrum()

icft = InverseCFT(cft.compute_cft(), frequencies, t)
x_rec = icft.reconstruct()

plt.plot(t, composite.values(), label="Original")
plt.plot(t, x_rec, '--', label="Reconstructed")
plt.legend()
plt.title("Reconstruction using ICFT")
plt.show()
