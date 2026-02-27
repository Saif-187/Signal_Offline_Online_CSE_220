from email.mime import audio
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
from discrete_framework import DFTAnalyzer, DiscreteSignal, FFTAnalyzer

class AudioEqualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("DFT Audio Equalizer")
        
        self.samplerate = 0
        self.original_audio = None
        self.processed_audio = None
        
        # --- UI Layout ---
        top_frame = tk.Frame(root)
        top_frame.pack(pady=10)
        
        tk.Button(top_frame, text="Load WAV File", command=self.load_file).pack(side=tk.LEFT, padx=10)
        tk.Button(top_frame, text="Process & Play", command=self.process_and_play).pack(side=tk.LEFT, padx=10)
        tk.Button(top_frame, text="Stop Audio", command=sd.stop).pack(side=tk.LEFT, padx=10)
        
        # Toggle Switch
        control_frame = tk.Frame(root)
        control_frame.pack(pady=5)
        self.use_fft = tk.BooleanVar(value=False)
        tk.Label(control_frame, text="Algorithm: ").pack(side=tk.LEFT)
        tk.Radiobutton(control_frame, text="DFT (Slow)", variable=self.use_fft, value=False).pack(side=tk.LEFT)
        tk.Radiobutton(control_frame, text="FFT (Fast)", variable=self.use_fft, value=True).pack(side=tk.LEFT)

        # Equalizer Sliders
        self.slider_frame = tk.Frame(root)
        self.slider_frame.pack(pady=20, padx=20)
        
        self.sliders = []
        labels = ["Low", "Low-Mid", "Mid", "High-Mid", "High"]
        for i in range(5):
            frame = tk.Frame(self.slider_frame)
            frame.pack(side=tk.LEFT, padx=5)
            tk.Label(frame, text=labels[i], font=("Arial", 8)).pack()
            slider = tk.Scale(frame, from_=2.0, to=0.0, resolution=0.1, length=150, orient=tk.VERTICAL)
            slider.set(1.0)
            slider.pack()
            self.sliders.append(slider)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            try:
                self.samplerate, data = wav.read(file_path)
                
                # Normalize to float [-1, 1]
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0
                elif data.dtype == np.uint8:
                    data = (data.astype(np.float32) - 128.0) / 128.0
                
                # If already float, just ensure float32
                if data.dtype != np.float32:
                    data = data.astype(np.float32)

                # Convert to mono if stereo
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                
                self.original_audio = data
                self.processed_audio = None
                duration = len(data) / self.samplerate
                print(f"Loaded: {len(data)} samples, {self.samplerate} Hz, {duration:.1f}s")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load file: {e}")

    def process_and_play(self):
        if self.original_audio is None:
            messagebox.showwarning("Warning", "Please load a WAV file first.")
            return
        
        print("Starting processing...")
        # Get Slider Values
        gains = [s.get() for s in self.sliders]
        
        # TODO: Implement the chunking, FFT, filtering, IFFT, and overlap-add here.
        
        # For starter code, we just play the original audio so the button "works"
        # In the final version, this should play self.processed_audio
        if self.use_fft.get():
            analyzer = FFTAnalyzer()
            chunk_size = 1024  
        else:
            analyzer = DFTAnalyzer()
            chunk_size = 256
        hop_size = chunk_size // 2 
        audio = self.original_audio
        total_samples = len(audio)
        output_audio = np.zeros(total_samples + chunk_size)
        sum_windows = np.zeros(total_samples + chunk_size)
        window = np.hanning(chunk_size)
        for start in range(0, total_samples, hop_size):
            end = min(start + chunk_size, total_samples)
            chunk = audio[start:end]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            chunk_windowed = chunk * window
            chunk_complex = chunk_windowed.astype(np.complex128)
            signal = DiscreteSignal(chunk_complex)
            spectrum = analyzer.compute_dft(signal)
            filtered_spectrum = self.apply_equalizer(spectrum, gains, self.samplerate, chunk_size)
            time_domain = analyzer.compute_idft(filtered_spectrum)
            time_domain_real = np.real(time_domain)
            output_audio[start:start+chunk_size] += time_domain_real * window
            sum_windows[start:start+chunk_size] += window
        sum_windows[sum_windows < 0.01] = 1.0
        output_audio = output_audio[:total_samples] / sum_windows[:total_samples]
        max_val = np.max(np.abs(output_audio))
        if max_val > 1.0:
            output_audio = output_audio / max_val
    
        self.processed_audio = output_audio
    
        print("Processing complete. Playing audio...")
        sd.stop()
        sd.play(output_audio, self.samplerate)
    def apply_equalizer(self, spectrum, gains, samplerate, chunk_size):
    
        N = len(spectrum)
        freqs = np.fft.fftfreq(N, 1/samplerate)
        bands = [
            (0, 200),          
            (200, 600),         
            (600, 2400),        
            (2400, 6000),       
            (6000, samplerate/2) 
        ]
        filtered = spectrum.copy()
        for i, (low_freq, high_freq) in enumerate(bands):
            mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) < high_freq)
            filtered[mask] *= gains[i]
    
        return filtered

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioEqualizer(root)
    root.mainloop()