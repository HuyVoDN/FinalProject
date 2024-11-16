import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import ttk
import scipy.ndimage as ndimage

def create_circular_phantom(matrix_size, structures):
    phantom = np.zeros((matrix_size, matrix_size))
    center = matrix_size // 2
    
    # Create main cylinder
    y, x = np.ogrid[-center:matrix_size-center, -center:matrix_size-center]
    mask = x*x + y*y <= (matrix_size//3)**2
    phantom[mask] = 0.2
    
    # Add circular structures
    for pos_x, pos_y, radius, value in structures:
        y, x = np.ogrid[-center:matrix_size-center, -center:matrix_size-center]
        structure_mask = (x - pos_x)**2 + (y - pos_y)**2 <= radius**2
        phantom[structure_mask] = value
    
    return phantom

def create_rectangular_phantom(matrix_size, rect_params):
    phantom = np.zeros((matrix_size, matrix_size))
    x, y, width, height, value = rect_params
    phantom[y:y+height, x:x+width] = value
    return phantom

def create_head_phantom(matrix_size):
    phantom = np.zeros((matrix_size, matrix_size))
    center = matrix_size // 2
    
    # Create skull (outer circle)
    y, x = np.ogrid[-center:matrix_size-center, -center:matrix_size-center]
    skull_mask = x*x + y*y <= (matrix_size//2.5)**2
    phantom[skull_mask] = 0.2
    
    # Create brain matter (inner circle)
    brain_mask = x*x + y*y <= (matrix_size//3)**2
    phantom[brain_mask] = 0.4
    
    # Add ventricles (two small ellipses)
    for offset in [-20, 20]:
        y, x = np.ogrid[-center:matrix_size-center, -center+offset:matrix_size-center+offset]
        ventricle_mask = (x*x)/(100) + (y*y)/(400) <= 1
        phantom[ventricle_mask] = 0.1
    
    # Add some brain structures
    structures = [
        (0, -10, 15, 0.5),    # Thalamus
        (0, 30, 20, 0.3),     # Frontal lobe structure
        (-30, 0, 10, 0.6),    # Left temporal structure
        (30, 0, 10, 0.6)      # Right temporal structure
    ]
    
    for pos_x, pos_y, radius, value in structures:
        y, x = np.ogrid[-center:matrix_size-center, -center:matrix_size-center]
        structure_mask = (x - pos_x)**2 + (y - pos_y)**2 <= radius**2
        phantom[structure_mask] = value
    
    return phantom

def simulate_ct_scan(phantom, angles, num_detectors, detector_spacing, source_distance):
    matrix_size = phantom.shape[0]
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
        rotated = ndimage.rotate(
            np.tile(filtered_sinogram[i], (matrix_size, 1)),
            -theta,
            reshape=False
        )
        reconstruction += rotated
    
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
        
        self.detector_spacing = tk.Scale(
            self.param_frame,
            from_=0.1,
            to=2.0,
            resolution=0.1,
            label="Detector Spacing",
            orient=tk.HORIZONTAL
        )
        self.detector_spacing.grid(row=1, column=0)
        
        # Scan Parameters
        self.angle_step = tk.Scale(
            self.param_frame,
            from_=0.1,
            to=5.0,
            resolution=0.1,
            label="Angle Step (degrees)",
            orient=tk.HORIZONTAL
        )
        self.angle_step.grid(row=2, column=0)
        
        # Control Buttons
        self.run_button = ttk.Button(
            self.param_frame,
            text="Run Scan",
            command=self.run_scan
        )
        self.run_button.grid(row=3, column=0)
    
    def run_scan(self):
        # Implement scan execution
        pass
    
def main():
    # Initialize parameters
    matrix_size = 256
    structures = [
        (20, 20, 10, 0.8),
        (-30, 30, 15, 0.5),
        (0, 0, 20, 0.3)
    ]
    
    # Create phantom
    phantom = create_circular_phantom(matrix_size, structures)
    
    # Scanner settings
    angles = np.linspace(0, 180, 180)
    num_detectors = 256
    detector_spacing = 1
    source_distance = matrix_size
    
    # Run simulation
    sinogram = simulate_ct_scan(phantom, angles, num_detectors, detector_spacing, source_distance)
    reconstruction = reconstruct_image(sinogram, angles, matrix_size)
    
    # Start GUI
    root = tk.Tk()
    app = CTScannerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()