import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import ttk
import scipy.ndimage as ndimage

def simulate_ct_scan(phantom, angles, num_detectors, detector_spacing, source_distance, matrix_size):
    sinogram = np.zeros((len(angles), num_detectors))
    
    for i, theta in enumerate(angles):
        rotated = ndimage.rotate(phantom, theta, reshape=False)
        
        for j in range(num_detectors):
            detector_pos = (j - num_detectors//2) * detector_spacing
            ray_path = np.linspace(
                [-source_distance, detector_pos],
                [source_distance, detector_pos],
                num=matrix_size
            )
            sinogram[i, j] = np.sum(
                ndimage.map_coordinates(rotated, ray_path.T, order=1)
            ) * (2 * source_distance / matrix_size)
    
    return sinogram

def reconstruct_image(sinogram, angles, matrix_size):
    reconstruction = np.zeros((matrix_size, matrix_size))
    filtered_sinogram = np.apply_along_axis(
        lambda x: np.convolve(x, np.array([1, -2, 1]), mode='same'),
        axis=1,
        arr=sinogram
    )
    
    for i, theta in enumerate(angles):
        projection = np.tile(filtered_sinogram[i], (matrix_size, 1))
        projection_resized = np.resize(projection, (matrix_size, matrix_size))
        rotated = ndimage.rotate(projection_resized, -theta, reshape=False)
        reconstruction += rotated[:matrix_size, :matrix_size]
    
    return reconstruction / len(angles)

def analyze_image(image, roi1, roi2):
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    
    mask1 = (x - roi1[0])**2 + (y - roi1[1])**2 <= roi1[2]**2
    mask2 = (x - roi2[0])**2 + (y - roi2[1])**2 <= roi2[2]**2
    
    si1 = np.mean(image[mask1])
    si2 = np.mean(image[mask2])
    contrast = abs(si1 - si2) / ((si1 + si2) / 2)
    
    return si1, si2, contrast

def get_intensity_profile(image, start_point, end_point):
    num_points = 100
    x = np.linspace(start_point[0], end_point[0], num_points)
    y = np.linspace(start_point[1], end_point[1], num_points)
    
    return ndimage.map_coordinates(image, [y, x], order=1)

def calculate_image_difference(original, reconstructed):
    return np.abs(original - reconstructed)

class CTScannerGUI:
    def add_analysis_tools(self):
    # Signal Intensity Profile
        self.profile_frame = ttk.LabelFrame(self.master, text="Profile Analysis")
        self.profile_button = ttk.Button(
            self.profile_frame, 
            text="Generate Profile",
            command=self.generate_profile
            )
    
    # Contrast Analysis
        self.contrast_frame = ttk.LabelFrame(self.master, text="Contrast Analysis")
        self.roi_selector = ttk.Button(
            self.contrast_frame,
            text="Select ROIs",
            command=self.analyze_contrast
    )
    def __init__(self, master):
        self.master = master
        self.master.title("Virtual CT Scanner")
        
        # Scanner Parameters Frame
        self.param_frame = ttk.LabelFrame(master, text="Scanner Parameters")
        self.param_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Detector Parameters
        self.num_detectors = tk.Scale(
            self.param_frame,
            from_=50,
            to=200,
            label="Number of Detectors",
            orient=tk.HORIZONTAL
        )
        self.num_detectors.grid(row=0, column=0)
        
        # Add more parameters and visualization components
        self.run_button = tk.Button(master, text="Run Scan", command=self.run_scan)
        self.run_button.grid(row=1, column=0, padx=5, pady=5)
        
        self.result_label = tk.Label(master, text="")
        self.result_label.grid(row=2, column=0, padx=5, pady=5)
    def add_scanner_controls(self):
        self.detector_type = ttk.Combobox(
            self.param_frame,
            values=["Linear", "Arc"]
        )
        self.step_angle = ttk.Scale(
            self.param_frame,
            from_=0.1,
            to=5.0,
            orient="horizontal"
        )
    def run_scan(self):
        # Example parameters
        matrix_size = 256
        angles = np.linspace(0, 180, 180, endpoint=False)
        detector_spacing = 1.0
        source_distance = 100.0
        
        # Create a simple phantom
        phantom = np.ones((matrix_size, matrix_size))
        center = matrix_size // 2
        
        for offset in [-20, 20]:
            y, x = np.ogrid[-center:matrix_size-center, -center+offset:matrix_size-center+offset]
            ventricle_mask = (x*x)/(100) + (y*y)/(400) <= 1
            phantom[ventricle_mask] = 0.1
        
        structures = [
            (0, -10, 15, 0.5),
            (0, 30, 20, 0.3),
            (-30, 0, 10, 0.6),
            (30, 0, 10, 0.6)
        ]
        
        for pos_x, pos_y, radius, value in structures:
            y, x = np.ogrid[-center:matrix_size-center, -center:matrix_size-center]
            structure_mask = (x - pos_x)**2 + (y - pos_y)**2 <= radius**2
            phantom[structure_mask] = value
        
        num_detectors = self.num_detectors.get()
        sinogram = simulate_ct_scan(phantom, angles, num_detectors, detector_spacing, source_distance, matrix_size)
        reconstruction = reconstruct_image(sinogram, angles, matrix_size)
        
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.title("Phantom")
        plt.imshow(phantom, cmap='gray')
        
        plt.subplot(1, 3, 2)
        plt.title("Sinogram")
        plt.imshow(sinogram, cmap='gray', aspect='auto')
        
        plt.subplot(1, 3, 3)
        plt.title("Reconstruction")
        plt.imshow(reconstruction, cmap='gray')
        
        plt.show()
        
        self.result_label.config(text="Scan completed")

if __name__ == "__main__":
    root = tk.Tk()
    app = CTScannerGUI(master=root)
    root.mainloop()