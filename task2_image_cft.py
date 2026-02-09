import numpy as np
import matplotlib.pyplot as plt
from imageio import imread

# =====================================================
# Continuous Image Class
# =====================================================
class ContinuousImage:
    """
    Represents an image as a continuous 2D signal.
    """

    def __init__(self, image_path):
        self.image = imread(image_path, mode='L')
        self.image = self.image / np.max(self.image)

        # Define continuous spatial axes
        self.x = np.linspace(-1, 1, self.image.shape[1])
        self.y = np.linspace(-1, 1, self.image.shape[0])

    def show(self, title="Image"):
        plt.imshow(self.image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()


# =====================================================
# 2D Continuous Fourier Transform Class
# =====================================================
class CFT2D:
    """
    Computes 2D Continuous Fourier Transform
    using separability and numerical integration.
    """

    def __init__(self, image_obj:ContinuousImage):
        self.I = image_obj.image
        self.x = image_obj.x
        self.y = image_obj.y

    def compute_cft(self):
        """
        Compute real and imaginary parts of 2D CFT.
        """
        rows, cols = self.I.shape
        real = np.zeros((rows, cols))
        imag = np.zeros((rows, cols))
        ui=np.linspace(-1, 1, rows)
        vj=np.linspace(-1, 1, cols)
        for i in range(rows):
            for j in range(cols):
                integ_cos = np.cos(2*np.pi*(ui[i]*self.x[np.newaxis, :] + vj[j]*self.y[:, np.newaxis]))
                integ_sin = np.sin(2*np.pi*(ui[i]*self.x[np.newaxis, :] + vj[j]*self.y[:, np.newaxis]))

                temp_cos = np.trapezoid(integ_cos * self.I, self.x, axis=1)
                temp_sin = np.trapezoid(integ_sin * self.I, self.x, axis=1)
                real[i,j] = np.trapezoid(temp_cos, self.y)
                imag[i,j] = -np.trapezoid(temp_sin, self.y)
        return real, imag


    def plot_magnitude(self):
        """
        Plot log-scaled magnitude spectrum.
        """
        real, imag = self.compute_cft()
        magnitude = np.sqrt(real**2 + imag**2)
        plt.imshow(np.log(magnitude + 1e-10), cmap='inferno')
        plt.title("2D CFT Magnitude Spectrum")
        plt.axis('off')
        plt.show()
        


# =====================================================
# Frequency Filtering
# =====================================================
class FrequencyFilter:
    def low_pass(self, real, imag, cutoff):
        rows, cols = real.shape
        cx, cy = rows//2, cols//2

        for i in range(rows):
            for j in range(cols):
                if np.sqrt((i-cx)**2 + (j-cy)**2) > cutoff:
                    real[i,j] = 0
                    imag[i,j] = 0
        return real, imag

# =====================================================
# Inverse 2D Continuous Fourier Transform
# =====================================================
class InverseCFT2D:
    """
    Reconstructs image from 2D frequency spectrum.
    """

    def __init__(self, real, imag, x, y):
        self.real=real 
        self.imag = imag
        self.x = x
        self.y = y

    def reconstruct(self):
        """
        Perform inverse 2D CFT using numerical integration.
        """
        rows, cols = self.real.shape
        output = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                integ_cos = np.cos(2*np.pi*(i*self.x[np.newaxis, :] + j*self.y[:, np.newaxis]))
                integ_sin = np.sin(2*np.pi*(i*self.x[np.newaxis, :] + j*self.y[:, np.newaxis]))

                temp_cos = np.trapezoid(integ_cos * self.real, self.x, axis=1)
                temp_sin = np.trapezoid(integ_sin * self.imag, self.x, axis=1)
                output[i,j] = np.trapezoid(temp_cos - temp_sin, self.y)
        return output

# =====================================================
# Main Execution (Task 2)
# =====================================================
img = ContinuousImage("noisy_image.png")
img.show("Original Image")

cft2d = CFT2D(img)
real, imag = cft2d.compute_cft()
cft2d.plot_magnitude()

filt = FrequencyFilter()
real_f, imag_f = filt.low_pass(real, imag, cutoff=40)

icft2d = InverseCFT2D(real_f, imag_f, img.x, img.y)
denoised = icft2d.reconstruct()

plt.imshow(denoised, cmap='gray')
plt.title("Reconstructed (Denoised) Image")
plt.axis('off')
plt.show()
