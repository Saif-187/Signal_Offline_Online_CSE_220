import numpy as np

class DiscreteSignal:
    """
    Represents a discrete-time signal.
    """
    def __init__(self, data):
        # Ensure data is a numpy array, potentially complex
        self.data = np.array(data, dtype=np.complex128)

    def __len__(self):
        return len(self.data)
        
    def pad(self, new_length):
        """
        Zero-pad or truncate signal to new_length.
        Returns a new DiscreteSignal object.
        """
        # TODO: Implement padding logic
        # Placeholder return to prevent crash
        length=len(self)
        if new_length < length:
            return DiscreteSignal(self.data[:new_length])
        elif new_length > length:
            data = np.zeros(new_length, dtype=np.complex128)
            data[:length] = self.data
            return DiscreteSignal(data)
        else:
            return DiscreteSignal(self.data.copy())
        #return DiscreteSignal(np.zeros(new_length))

    def interpolate(self, new_length):
        """
        Resample signal to new_length using linear interpolation.
        Required for Task 4 (Drawing App).
        """
        # TODO: Implement interpolation logic
        if new_length == len(self.data):
            return DiscreteSignal(self.data.copy())
        old=np.arange(len(self.data))
        new=np.linspace(0, len(self.data)-1, new_length)
        real=np.interp(new,old,self.data.real)
        imag=np.interp(new,old,self.data.imag)
        return DiscreteSignal(real+1j*imag)
        #return DiscreteSignal(np.zeros(new_length))


class DFTAnalyzer:
    """
    Performs Discrete Fourier Transform using O(N^2) method.
    """
    def compute_dft(self, signal: DiscreteSignal):
        """
        Compute DFT using naive summation.
        Returns: numpy array of complex frequency coefficients.
        """
        N = len(signal)
        # TODO: Implement Naive DFT equation
        # Placeholder: Return zeros so UI doesn't crash
        x=signal.data
        Output_Array=np.zeros(N, dtype=np.complex128)
        for k in range(N):
            sum=0
            for n in range(N):
                expo=np.exp(-2j*np.pi*k*n/N)
                sum+=x[n]*expo
            Output_Array[k]=sum
        return Output_Array
        #return np.zeros(N, dtype=np.complex128)

    def compute_idft(self, spectrum):
        """
        Compute Inverse DFT using naive summation.
        Returns: numpy array (time-domain samples).
        """
        # TODO: Implement Naive IDFT equation
        N=len(spectrum)
        x=np.zeros(N, dtype=np.complex128)
        for n in range(N):
            sum=0
            for k in range(N):
                expo=np.exp(2j*np.pi*k*n/N)
                sum+=spectrum[k]*expo
            x[n]=sum/N
        return x
        #return np.zeros(len(spectrum), dtype=np.complex128)
class FFTAnalyzer(DFTAnalyzer):
    def compute_dft(self, signal: DiscreteSignal):
        return self.fft(signal.data)
    def compute_idft(self, spectrum):
        return self.ifft(spectrum)
    def compute_fft(self, x):
        N = len(x)
        if N <= 1:
            return x
        if N & (N - 1) != 0:
            next_pow2 = 1
            while next_pow2 < N:
                next_pow2 <<= 1
            x = np.pad(x, (0, next_pow2 - N), mode='constant')
            N = next_pow2
        return self.recursive_fft(x)
    def recursive_fft(self, x):
        N = len(x)
        if N <= 1:
            return x
        even = self.recursive_fft(x[0::2])
        odd = self.recursive_fft(x[1::2])
        T = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([even + T[:N // 2] * odd, even + T[N // 2:] * odd])
    def ifft(self, X):
        N = len(X)
        x_conj = self.fft(np.conj(X))
        return np.conj(x_conj) / N