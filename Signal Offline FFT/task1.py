import tkinter as tk
import numpy as np
import math
from discrete_framework import DiscreteSignal, DFTAnalyzer, FFTAnalyzer

class DoodlingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fourier Epicycles Doodler")
        
        # --- UI Layout ---
        self.canvas = tk.Canvas(root, width=800, height=600, bg="white")
        self.canvas.pack()
        
        control_frame = tk.Frame(root)
        control_frame.pack(pady=10)
        
        # Buttons
        tk.Button(control_frame, text="Clear Canvas", command=self.clear).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Draw Epicycles", command=self.run_transform).pack(side=tk.LEFT, padx=5)
        
        # Toggle Switch (Radio Buttons)
        self.use_fft = tk.BooleanVar(value=False)
        tk.Label(control_frame, text=" |  Algorithm: ").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(control_frame, text="Naive DFT", variable=self.use_fft, value=False).pack(side=tk.LEFT)
        tk.Radiobutton(control_frame, text="FFT", variable=self.use_fft, value=True).pack(side=tk.LEFT)

        # State Variables
        self.points = []
        self.drawing = False
        self.fourier_coeffs = None
        self.is_animating = False
        self.after_id = None

        # Bindings
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)

    def start_draw(self, event):
        self.is_animating = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.canvas.delete("all")
        self.points = []
        self.drawing = True

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.points.append((x, y))
            r = 2
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")

    def end_draw(self, event):
        self.drawing = False

    def clear(self):
        self.canvas.delete("all")
        self.points = []
        self.is_animating = False
        if self.after_id:
            self.root.after_cancel(self.after_id)

    def draw_epicycle(self, x, y, radius):
        """
        Helper method for students to draw a circle (epicycle).
        x, y: Center coordinates
        radius: Radius of the circle
        """
        self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, outline="blue", tags="epicycle")

    def run_transform(self):
        if not self.points: return
        
        # TODO: Implementation
        # 1. Convert (x,y) points to Complex Signal
        # 2. Select Algorithm 
        # 3. Compute Transform
        x, y = [], []
        for point in self.points:
            x.append(point[0])
            y.append(point[1])
        x=np.array(x)
        y=np.array(y)
        comp= x + 1j * y
        signal= DiscreteSignal(comp)
        if self.use_fft.get():
            N = len(signal)
            next_pow2 = 1
            while next_pow2 < N:
                next_pow2 <<= 1
            
            if next_pow2 != N:
                signal = signal.interpolate(next_pow2)
        if self.use_fft.get():
            analyzer = FFTAnalyzer()
        else:
            analyzer = DFTAnalyzer()
        self.fourier_coeffs = analyzer.compute_dft(signal)
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        self.center_offset = (mean_x, mean_y)
        self.num_frames = len(signal)
        self.time_step = 0
        self.animate_epicycles(self.center_offset)
        # print("Transform logic needed.")
        # self.animate_epicycles(mean_point)

    def animate_epicycles(self, center_offset):
        self.is_animating = True
        self.time_step = 0
        # self.num_frames = ...
        
        self.center_offset = center_offset
        self.update_frame()

    def update_frame(self):
        if not self.is_animating: return
        
        self.canvas.delete("epicycle") 
        self.canvas.delete("tip")
        # TODO: Implementation
        # 1. Calculate the current time 't' based on self.time_step
        # 2. Reconstruct the signal value 'z' at time 't' 
        # 3. Draw the epicycles:
        # 4. Draw the tips
        N = len(self.fourier_coeffs)
        if N == 0:
            return
        t=self.time_step / self.num_frames

        x,y=self.center_offset
        magnitude=np.abs(self.fourier_coeffs)
        indices = np.argsort(-np.array(magnitude)) 

        for k_idx, k in enumerate(indices):
            coeff = self.fourier_coeffs[k]
            magnitude = np.abs(coeff)
            phase = np.angle(coeff)
            rotation = np.exp(1j * 2 * np.pi * k * t)
            vector = coeff * rotation / N 
            
            if k_idx < 20: 
                self.draw_epicycle(x, y, magnitude/N)
            
            x += np.real(vector)
            y += np.imag(vector)
        
        r = 4
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="red", outline="red", tags="tip")
        if hasattr(self, 'prev_x') and hasattr(self, 'prev_y'):
            self.canvas.create_line(self.prev_x, self.prev_y, x, y, fill="green", width=2, tags="epicycle")
        
        self.prev_x, self.prev_y = x, y
        self.time_step = (self.time_step + 1) % self.num_frames
        # Loop animation
        self.after_id = self.root.after(50, self.update_frame)

        
if __name__ == "__main__":
    root = tk.Tk()
    app = DoodlingApp(root)
    root.mainloop()