import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
from discrete_framework import DFTAnalyzer, FFTAnalyzer

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
            while chunk_size < 1024:  
                chunk_size <<= 1
        else:
            analyzer = DFTAnalyzer()
            chunk_size = 256
        audio = self.original_audio
        total_samples = len(audio)
        processed_chunks = []
        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)
            chunk = audio[start:end]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
            window = np.hanning(len(chunk))
            chunk_windowed = chunk * window
            chunk_complex = chunk_windowed.astype(np.complex128)
            signal = type('Signal', (), {'data': chunk_complex}) 
            spectrum = analyzer.compute_dft(signal)
            filtered_spectrum = self.apply_equalizer(spectrum, gains, self.samplerate, chunk_size)
            time_domain = analyzer.compute_idft(filtered_spectrum)
            time_domain_real = np.real(time_domain)
            time_domain_real = time_domain_real * window
            processed_chunks.append(time_domain_real[:end-start])
        output_audio = np.concatenate(processed_chunks)[:total_samples]
        max_val = np.max(np.abs(output_audio))
        if max_val > 1.0:
            output_audio = output_audio / max_val
        
        self.processed_audio = output_audio
        
        print("Processing complete. Playing audio...")
    
        sd.stop()
        default_output = sd.default.device[0]
        sd.play(output_audio, self.samplerate, device=default_output)

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioEqualizer(root)
    root.mainloop()