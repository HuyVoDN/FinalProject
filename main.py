import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from skimage.draw import disk, rectangle
import random

class VirtualCTScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Virtual CT Scanner")
        self.root.geometry("800x600")

        # Phantom selection
        ttk.Label(root, text="Select Phantom:").grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.phantom_var = tk.StringVar(value="Cylinder")
        self.phantom_dropdown = ttk.Combobox(root, textvariable=self.phantom_var, values=["Cylinder", "Head"], state="readonly")
        self.phantom_dropdown.grid(row=0, column=1, padx=10, pady=5, sticky='w')
        self.phantom_dropdown.bind("<<ComboboxSelected>>", self.update_phantom_controls)

        # Matrix size
        ttk.Label(root, text="Matrix Size:").grid(row=1, column=0, padx=10, pady=5, sticky='w')
        self.matrix_size_var = tk.IntVar(value=400)  # Set default value to 400
        self.matrix_size_spinner = ttk.Spinbox(root, from_=400, to=512, textvariable=self.matrix_size_var)  # Ensure minimum value is 400
        self.matrix_size_spinner.grid(row=1, column=1, padx=10, pady=5, sticky='w')

        # Structure values and dimensions for cylinder
        self.cylinder_controls = []
        self.add_cylinder_controls(root)

        # Structure values and dimensions for head
        self.head_controls = []
        self.add_head_controls(root)

        # Data acquisition parameters
        ttk.Label(root, text="Number of Detectors:").grid(row=2, column=0, padx=10, pady=5, sticky='w')
        self.num_detectors_var = tk.IntVar(value=180)
        self.num_detectors_spinner = ttk.Spinbox(root, from_=1, to=360, textvariable=self.num_detectors_var)
        self.num_detectors_spinner.grid(row=2, column=1, padx=10, pady=5, sticky='w')

        ttk.Label(root, text="Detector Spacing:").grid(row=3, column=0, padx=10, pady=5, sticky='w')
        self.detector_spacing_var = tk.DoubleVar(value=1.0)
        self.detector_spacing_spinner = ttk.Spinbox(root, from_=0.1, to=10.0, increment=0.1, textvariable=self.detector_spacing_var)
        self.detector_spacing_spinner.grid(row=3, column=1, padx=10, pady=5, sticky='w')

        ttk.Label(root, text="Source Distance:").grid(row=4, column=0, padx=10, pady=5, sticky='w')
        self.source_distance_var = tk.DoubleVar(value=100.0)
        self.source_distance_spinner = ttk.Spinbox(root, from_=10.0, to=500.0, increment=10.0, textvariable=self.source_distance_var)
        self.source_distance_spinner.grid(row=4, column=1, padx=10, pady=5, sticky='w')

        ttk.Label(root, text="Step Angle:").grid(row=5, column=0, padx=10, pady=5, sticky='w')
        self.step_angle_var = tk.DoubleVar(value=1.0)
        self.step_angle_spinner = ttk.Spinbox(root, from_=0.1, to=10.0, increment=0.1, textvariable=self.step_angle_var)
        self.step_angle_spinner.grid(row=5, column=1, padx=10, pady=5, sticky='w')

        # Buttons and instructions
        ttk.Button(root, text="Reset", command=self.reset_app).grid(row=7, column=0, padx=10, pady=10, sticky='w')
        ttk.Label(root, text="Reset all parameters and clear images").grid(row=7, column=1, padx=10, pady=10, sticky='w')

        ttk.Button(root, text="Acquire Data", command=self.acquire_data).grid(row=8, column=0, padx=10, pady=10, sticky='w')
        ttk.Label(root, text="Generate phantom and simulate acquisition").grid(row=8, column=1, padx=10, pady=10, sticky='w')

        ttk.Button(root, text="Reconstruct Image", command=self.reconstruct_image).grid(row=9, column=0, padx=10, pady=10, sticky='w')
        ttk.Label(root, text="Reconstruct image from sinogram").grid(row=9, column=1, padx=10, pady=10, sticky='w')

        ttk.Button(root, text="Export Data", command=self.export_sinogram).grid(row=10, column=0, padx=10, pady=10, sticky='w')
        ttk.Label(root, text="Save sinogram data to a file").grid(row=10, column=1, padx=10, pady=10, sticky='w')

        ttk.Button(root, text="SI & Contrast", command=self.analyze_si_contrast).grid(row=11, column=0, padx=10, pady=10, sticky='w')
        ttk.Label(root, text="Analyze signal intensity and contrast").grid(row=11, column=1, padx=10, pady=10, sticky='w')

        ttk.Button(root, text="Image Difference", command=self.analyze_image_difference).grid(row=12, column=0, padx=10, pady=10, sticky='w')
        ttk.Label(root, text="Compute and display image difference").grid(row=12, column=1, padx=10, pady=10, sticky='w')

        ttk.Button(root, text="SI Profiles", command=self.analyze_si_profiles).grid(row=13, column=0, padx=10, pady=10, sticky='w')
        ttk.Label(root, text="Generate and display SI profiles").grid(row=13, column=1, padx=10, pady=10, sticky='w')

        ttk.Button(root, text="Compare & Contrast", command=self.compare_and_contrast).grid(row=14, column=0, padx=10, pady=10, sticky='w')
        ttk.Label(root, text="Compare original and reconstructed images").grid(row=14, column=1, padx=10, pady=10, sticky='w')

        # Initialize variables for phantom, sinogram, and reconstructed image
        self.original_phantom = None
        self.phantom = None
        self.sinogram = None
        self.reconstructed_image = None

        self.update_phantom_controls()

    def add_cylinder_controls(self, root):
        self.cylinder_controls.append(ttk.Label(root, text="Rectangle Width:"))
        self.cylinder_controls[-1].grid(row=3, column=0, padx=10, pady=5, sticky='w')
        self.rect_width_var = tk.DoubleVar(value=0.1)
        self.rect_width_spinner = ttk.Spinbox(root, from_=0, to=0.5, increment=0.01, textvariable=self.rect_width_var)
        self.rect_width_spinner.grid(row=3, column=1, padx=10, pady=5, sticky='w')
        self.cylinder_controls.append(self.rect_width_spinner)

        self.cylinder_controls.append(ttk.Label(root, text="Rectangle Height:"))
        self.cylinder_controls[-1].grid(row=4, column=0, padx=10, pady=5, sticky='w')
        self.rect_height_var = tk.DoubleVar(value=0.2)
        self.rect_height_spinner = ttk.Spinbox(root, from_=0, to=0.5, increment=0.01, textvariable=self.rect_height_var)
        self.rect_height_spinner.grid(row=4, column=1, padx=10, pady=5, sticky='w')
        self.cylinder_controls.append(self.rect_height_spinner)

    def add_head_controls(self, root):
        self.head_controls.append(ttk.Label(root, text="Number of Circles:"))
        self.head_controls[-1].grid(row=3, column=0, padx=10, pady=5, sticky='w')
        self.num_circles_var = tk.IntVar(value=3)
        self.num_circles_spinner = ttk.Spinbox(root, from_=1, to=10, textvariable=self.num_circles_var)
        self.num_circles_spinner.grid(row=3, column=1, padx=10, pady=5, sticky='w')
        self.head_controls.append(self.num_circles_spinner)

        self.head_controls.append(ttk.Label(root, text="Circle Radii:"))
        self.head_controls[-1].grid(row=4, column=0, padx=10, pady=5, sticky='w')
        self.circle_radii_var = tk.StringVar(value="0.1,0.2,0.3")
        self.circle_radii_entry = ttk.Entry(root, textvariable=self.circle_radii_var)
        self.circle_radii_entry.grid(row=4, column=1, padx=10, pady=5, sticky='w')
        self.head_controls.append(self.circle_radii_entry)

        self.head_controls.append(ttk.Label(root, text="Circle Intensities:"))
        self.head_controls[-1].grid(row=5, column=0, padx=10, pady=5, sticky='w')
        self.circle_intensities_var = tk.StringVar(value="1,0.8,0.6")
        self.circle_intensities_entry = ttk.Entry(root, textvariable=self.circle_intensities_var)
        self.circle_intensities_entry.grid(row=5, column=1, padx=10, pady=5, sticky='w')
        self.head_controls.append(self.circle_intensities_entry)

    def update_phantom_controls(self, event=None):
        phantom_type = self.phantom_var.get()
        if (phantom_type == "Cylinder"):
            for control in self.cylinder_controls:
                control.grid()
            for control in self.head_controls:
                control.grid_remove()
        else:
            for control in self.cylinder_controls:
                control.grid_remove()
            for control in self.head_controls:
                control.grid()

    def reset_app(self):
        self.original_phantom = None
        self.phantom = None
        self.sinogram = None
        self.reconstructed_image = None
        self.phantom_var.set("Cylinder")
        self.matrix_size_var.set(400)  # Set default value to 400
        self.rect_width_var.set(0.1)
        self.rect_height_var.set(0.2)
        self.num_circles_var.set(3)
        self.circle_radii_var.set("0.1,0.2,0.3")
        self.circle_intensities_var.set("1,0.8,0.6")
        self.update_phantom_controls()

    def acquire_data(self):
        matrix_size = self.matrix_size_var.get()
        phantom_type = self.phantom_var.get()

        if phantom_type == "Cylinder":
            self.phantom = self.generate_cylinder_phantom(matrix_size)
        else:
            self.phantom = self.generate_head_phantom(matrix_size)

        num_detectors = self.num_detectors_var.get()
        detector_spacing = self.detector_spacing_var.get()
        source_distance = self.source_distance_var.get()
        step_angle = self.step_angle_var.get()

        angles = np.arange(0, 180, step_angle)
        self.sinogram = radon(self.phantom, angles, circle=True)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title("Phantom")
        plt.imshow(self.phantom, cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title("Sinogram")
        plt.imshow(self.sinogram, cmap='gray', aspect='auto')

        plt.show()

        # Print acquisition parameters
        print(f"Acquisition Parameters:\n"
              f"Number of Detectors: {num_detectors}\n"
              f"Detector Spacing: {detector_spacing}\n"
              f"Source Distance: {source_distance}\n"
              f"Step Angle: {step_angle}\n")

    def generate_cylinder_phantom(self, matrix_size):
        phantom = np.zeros((matrix_size, matrix_size))
        rr, cc = rectangle(start=(matrix_size//4, matrix_size//4), extent=(self.rect_height_var.get()*matrix_size, self.rect_width_var.get()*matrix_size))
        phantom[rr, cc] = 1
        return phantom

    def generate_head_phantom(self, matrix_size):
        phantom = np.zeros((matrix_size, matrix_size))
        num_circles = self.num_circles_var.get()
        radii = list(map(float, self.circle_radii_var.get().split(',')))
        intensities = list(map(float, self.circle_intensities_var.get().split(',')))

        for i in range(num_circles):
            rr, cc = disk((random.randint(0, matrix_size), random.randint(0, matrix_size)), radii[i]*matrix_size)
            phantom[rr, cc] = intensities[i]

        return phantom

    def reconstruct_image(self):
        if self.sinogram is None:
            messagebox.showerror("Error", "No sinogram data available. Please acquire data first.")
            return

        angles = np.arange(0, 180, self.step_angle_var.get())
        self.reconstructed_image = iradon(self.sinogram, angles, circle=True)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title("Sinogram")
        plt.imshow(self.sinogram, cmap='gray', aspect='auto')

        plt.subplot(1, 2, 2)
        plt.title("Reconstructed Image")
        plt.imshow(self.reconstructed_image, cmap='gray')

        plt.show()

    def export_sinogram(self):
        if self.sinogram is None:
            messagebox.showerror("Error", "No sinogram data available. Please acquire data first.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("NumPy files", "*.npy")])
        if file_path:
            np.save(file_path, self.sinogram)
            messagebox.showinfo("Success", f"Sinogram data saved to {file_path}")

    def analyze_si_contrast(self):
        if self.reconstructed_image is None:
            messagebox.showerror("Error", "No reconstructed image available. Please reconstruct the image first.")
            return

        # Example analysis: calculate mean intensity and contrast
        mean_intensity = np.mean(self.reconstructed_image)
        contrast = np.max(self.reconstructed_image) - np.min(self.reconstructed_image)

        messagebox.showinfo("SI & Contrast", f"Mean Intensity: {mean_intensity}\nContrast: {contrast}")

    def analyze_image_difference(self):
        if self.phantom is None or self.reconstructed_image is None:
            messagebox.showerror("Error", "No phantom or reconstructed image available. Please acquire data and reconstruct the image first.")
            return

        difference = np.abs(self.phantom - self.reconstructed_image)

        plt.figure()
        plt.title("Image Difference")
        plt.imshow(difference, cmap='gray')
        plt.show()

    def analyze_si_profiles(self):
        if self.reconstructed_image is None:
            messagebox.showerror("Error", "No reconstructed image available. Please reconstruct the image first.")
            return

        # Example SI profile: horizontal and vertical profiles through the center
        center = self.reconstructed_image.shape[0] // 2
        horizontal_profile = self.reconstructed_image[center, :]
        vertical_profile = self.reconstructed_image[:, center]

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title("Horizontal SI Profile")
        plt.plot(horizontal_profile)

        plt.subplot(1, 2, 2)
        plt.title("Vertical SI Profile")
        plt.plot(vertical_profile)

        plt.show()

    def compare_and_contrast(self):
        if self.phantom is None or self.reconstructed_image is None:
            messagebox.showerror("Error", "No phantom or reconstructed image available. Please acquire data and reconstruct the image first.")
            return

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title("Original Phantom")
        plt.imshow(self.phantom, cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title("Reconstructed Image")
        plt.imshow(self.reconstructed_image, cmap='gray')

        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = VirtualCTScannerApp(root)
    root.mainloop()