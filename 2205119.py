import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

class FourierSeries:
    def __init__(self, func, L, terms=10):
        """
        Initialize the FourierSeries class with a target function, period L, and number of terms.
        """
        self.func = func
        self.L = L
        self.terms = terms
        # Implement this method

    def calculate_a0(self, N=1000):
        """
        Step 1: Compute the a0 coefficient using numerical integration.
        a0 = (1/2L) * integral(f(x), -L, L)
        """
        # TODO: Implement numerical integration for a0
        spaced_points=np.linspace(-self.L, self.L, N)
        func_values=self.func(spaced_points)
        integral_value=np.trapezoid(func_values, spaced_points)
        a0 = (1/(self.L)) * integral_value
        return a0

    def calculate_an(self, n, N=1000):
        """
        Step 2: Compute the an coefficient for the nth cosine term.
        an = (1/L) * integral(f(x) * cos(n*pi*x/L), -L, L)
        """
        # TODO: Implement numerical integration for an
        spaced_points=np.linspace(-self.L, self.L, N)
        func_values=self.func(spaced_points)
        cos_values=np.cos(n*np.pi*spaced_points/self.L)
        integral_value=np.trapezoid(func_values*cos_values, spaced_points)
        an = (1/self.L) * integral_value
        return an

    def calculate_bn(self, n, N=1000):
        """
        Step 3: Compute the bn coefficient for the nth sine term.
        bn = (1/L) * integral(f(x) * sin(n*pi*x/L), -L, L)
        """
        # TODO: Implement numerical integration for bn
        spaced_points=np.linspace(-self.L, self.L, N)
        func_values=self.func(spaced_points)
        sin_values=np.sin(n*np.pi*spaced_points/self.L)
        integral_value=np.trapezoid(func_values*sin_values, spaced_points)
        bn = (1/self.L) * integral_value
        return bn

    def approximate(self, x):
        """
        Step 4: Use the calculated coefficients to build the Fourier series approximation.
        f(x) approx = a0/2 + sum(an*cos + bn*sin)
        """
        # TODO: Implement the Fourier series summation
        approx=self.calculate_a0()/2
        for n in range(1, self.terms + 1):
            an=self.calculate_an(n)
            bn=self.calculate_bn(n)
            approx += an * np.cos(n * np.pi * x / self.L) + bn * np.sin(n * np.pi * x / self.L)
        return approx

    def plot(self, ax, wave_type="square"):
        """
        Step 5: Plot the original function and its Fourier series approximation.
        Now plots multiple periods.
        """
        x = None #Implement this
        if wave_type == "cubic":
            x = np.linspace(-6, 6, 4000)
        else:
            x = np.linspace(-4*np.pi, 4*np.pi, 4000)

        # Compute original function values
        original = self.func(x)
        
        # Compute Fourier series approximation
        approximation = self.approximate(x)

        # Clear axis and Plotting
        ax.clear()
        ax.plot(x, original, label="Original Function", color="blue", alpha=0.5)
        ax.plot(x, approximation, label=f"Fourier Series (N={self.terms})", color="red", linestyle="--")
        
        # Dynamic Y-limits to ensure full view is seen for all wave types
        if wave_type == "sawtooth":
            # Sawtooth goes from -pi to +pi
            ax.set_ylim(-3.5, 3.5) 
        elif wave_type == "cubic":
            # Cubic x^3 on -1 to 1 ranges from -1 to 1.
            ax.set_ylim(-1.5, 1.5)
        elif wave_type == "pulse":
            ax.set_ylim(-0.5, 1.5)
        else:
            # Square, Triangle are roughly +/- 1
            ax.set_ylim(-1.5, 1.5)
        # Set X-limits to show multiple periods
        if wave_type == "cubic":
            ax.set_xlim(-6, 6)
        else:
            ax.set_xlim(-4 * np.pi, 4 * np.pi)
        
        ax.legend(loc='upper right')
        ax.grid(True)
        ax.set_title(f"Fourier Series Approximation: {wave_type.replace('_', ' ').title()}")


def target_function(x, function_type="square"):
    """
    Defines target functions.
    """
    if function_type == "square":
        return np.where(np.sin(x) >= 0, 1, -1)
    elif function_type == "sawtooth":
        x1=x+np.pi
        x1=x1%(2*np.pi)
        x1-=np.pi
        return x1
    elif function_type == "triangle":
        return np.arcsin(np.sin(x))
    elif function_type == "cubic":
        x_mod = ((x + 1) % 2) - 1
        return x_mod*x_mod*x_mod
    elif function_type == "pulse":
        x_mod = x % (2*np.pi)
        return np.where(x_mod < 0.175, 1.0, 0.0)
    else:
        raise ValueError("Invalid function_type.")
def get_half_period(wave_type):
    """
    Get the half-period L for different wave types.
    """
    if wave_type == "cubic":
        return 1 
    else:
        return np.pi


# Example of using these functions in the FourierSeries class with Sliders
if __name__ == "__main__":
    initial_terms = 1  # Start with 1 term
    initial_wave = "square"
    L = get_half_period(initial_wave)  # Half-period for initial function

    # Create the plot figure and axis
    fig_plot, ax_plot = plt.subplots(figsize=(10, 6))

    # Create the widgets figure
    fig_widgets = plt.figure(figsize=(8, 4))
    current_func = lambda x: target_function(x, initial_wave)
    fs = FourierSeries(current_func, L, initial_terms)
    # Pass the wave type to plot for correct axis scaling
    fs.plot(ax_plot, initial_wave)



    # 1. Radio Buttons (Positioned at the bottom left)
    ax_radio = fig_widgets.add_axes([0.05, 0.1, 0.15, 0.3], facecolor='#f0f0f0')
    radio = RadioButtons(ax_radio, ('square', 'sawtooth', 'triangle', 'cubic', 'pulse'))

    # 2. Slider Axes (Positioned above the radio buttons)
    ax_n = fig_widgets.add_axes([0.25, 0.6, 0.6, 0.1])

    # Slider for N (Number of terms)
    slider_n = Slider(
        ax=ax_n,
        label='Harmonics (N)',
        valmin=1,
        valmax=500,
        valinit=initial_terms,
        valstep=1
    )

    # --- Update Logic ---

    def update(val):
        """Callback when slider moves."""
        n = int(slider_n.val)
        wave_type = radio.value_selected
        
        # Update the FourierSeries object
        fs.terms = n
        fs.L = get_half_period(wave_type)
        fs.func = lambda x: target_function(x, wave_type)
        
        # Re-plot (Pass wave_type to handle axis scaling)
        fs.plot(ax_plot, wave_type)
        fig_plot.canvas.draw_idle()

    def change_wave(label):
        """Callback when radio button changes."""
        wave_type = label
        
        # Update the FourierSeries object
        fs.terms = int(slider_n.val) # Keep current N
        fs.L = get_half_period(wave_type)
        fs.func = lambda x: target_function(x, wave_type)
        
        # Re-plot (Pass wave_type to handle axis scaling)
        fs.plot(ax_plot, wave_type)
        fig_plot.canvas.draw_idle()

    # Connect widgets to functions
    slider_n.on_changed(update)
    radio.on_clicked(change_wave)

    plt.show()