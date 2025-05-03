import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageEnhance, ImageStat, ImageFilter
import numpy as np
import subprocess, os, config
from numba import njit
from helper import fs_dither, simple_threshold_rgb_ps1, simple_threshold_dither
class DitherApp:
    def __init__(self, root):
        self.root = root
        self.root.title(config.APP_NAME)
        self.root.geometry("1200x800")
        self.original_image = None
        self.dithered_image = None
        # Auto-render state
        self.auto_render_var = tk.BooleanVar(value=True)
        self.rgb_or_greyscale_var = tk.BooleanVar(value=True)
        # --- Layout ---
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        top_frame = tk.Frame(root)
        top_frame.grid(row=0, column=0, sticky='ew', padx=10, pady=5)
        top_frame.grid_columnconfigure(1, weight=1)
        top_frame.grid_columnconfigure(3, weight=1)
        top_frame.grid_columnconfigure(8, weight=1)

        btn_open = tk.Button(
            top_frame, text="Open Image", command=self.open_image,
            bg='lightgray', activebackground='blue', activeforeground='white'
        )
        btn_open.grid(row=0, column=0, padx=5, pady=2)

        btn_save = tk.Button(
            top_frame, text="Save Image", command=self.save_image,
            state=tk.DISABLED, bg='lightgray', activebackground='blue', activeforeground='white'
        )
        btn_save.grid(row=0, column=1, padx=5, pady=2, sticky='w')
        self.save_button = btn_save

        lbl_algo = tk.Label(top_frame, text="Algorithm:")
        lbl_algo.grid(row=0, column=2, padx=(10,5), pady=2, sticky='e')
        self.algorithm_var = tk.StringVar()
        self.algo_combobox = ttk.Combobox(
            top_frame, textvariable=self.algorithm_var,
            values=["Floyd-Steinberg", "Simple Threshold"], state='readonly', width=15
        )
        self.algo_combobox.current(0)
        self.algo_combobox.grid(row=0, column=3, padx=5, pady=2, sticky='w')

        btn_apply = tk.Button(
            top_frame, text="Apply Dither", command=self.apply_dither,
            bg='lightgray', activebackground='blue', activeforeground='white'
        )
        btn_apply.grid(row=0, column=7, padx=5, pady=2, sticky='e')

        # Add Auto-Render Checkbutton
        auto_render_check = ttk.Checkbutton(top_frame, text="Auto-Render", variable=self.auto_render_var)
        auto_render_check.grid(row=0, column=8, padx=5, pady=2, sticky='e')

        rgb_or_greyscale_check = ttk.Checkbutton(top_frame, text="Color", variable=self.rgb_or_greyscale_var, 
                                               command=lambda: self.update_rgb_or_greyscale_and_apply(None))
        rgb_or_greyscale_check.grid(row=0, column=9, padx=5, pady=2, sticky='e')

        image_frame = tk.Frame(root)
        image_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=5)
        image_frame.grid_rowconfigure(0, weight=1)
        image_frame.grid_columnconfigure(0, weight=1)
        image_frame.grid_columnconfigure(1, weight=1)

        frame_orig = tk.LabelFrame(image_frame, text="Original")
        frame_orig.grid(row=0, column=0, sticky='nsew', padx=5)
        frame_orig.grid_rowconfigure(0, weight=1)
        frame_orig.grid_columnconfigure(0, weight=1)
        frame_orig.grid_propagate(False)
        self.lbl_original = tk.Label(frame_orig)
        self.lbl_original.grid(sticky='nsew')

        frame_dith = tk.LabelFrame(image_frame, text="Dithered")
        frame_dith.grid(row=0, column=1, sticky='nsew', padx=5)
        frame_dith.grid_rowconfigure(0, weight=1)
        frame_dith.grid_columnconfigure(0, weight=1)
        frame_dith.grid_propagate(False)
        self.lbl_dithered = tk.Label(frame_dith)
        self.lbl_dithered.grid(sticky='nsew')

        # --- Row 1 Controls ---
        lbl_threshold = tk.Label(top_frame, text="Threshold:")
        lbl_threshold.grid(row=1, column=0, padx=5, pady=2, sticky='w')
        self.threshold_slider = ttk.Scale(top_frame, from_=1, to=254, orient=tk.HORIZONTAL, length=150, command=self.update_threshold_and_apply)
        self.threshold_slider.set(128)
        self.threshold_slider.grid(row=1, column=1, padx=2, pady=2, sticky='ew')
        self.lbl_thr_val = tk.Label(top_frame, text="128")
        self.lbl_thr_val.grid(row=1, column=2, padx=(0,10), pady=2, sticky='w')

        lbl_contrast = tk.Label(top_frame, text="Contrast:")
        lbl_contrast.grid(row=1, column=3, padx=(10,5), pady=2, sticky='w')
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.contrast_slider = ttk.Scale(top_frame, from_=0.1, to=5.0, orient=tk.HORIZONTAL, length=150, variable=self.contrast_var, command=self.update_contrast_and_apply)
        self.contrast_slider.grid(row=1, column=4, padx=2, pady=2, sticky='ew')
        self.lbl_con_val = tk.Label(top_frame, text="1.0")
        self.lbl_con_val.grid(row=1, column=5, padx=(0,10), pady=2, sticky='w')

        lbl_scale = tk.Label(top_frame, text="Pixel Scale:")
        lbl_scale.grid(row=1, column=6, padx=(10,5), pady=2, sticky='w')
        self.scale_var = tk.IntVar(value=1)
        self.scale_slider = ttk.Scale(top_frame, from_=1, to=8, orient=tk.HORIZONTAL, length=100, variable=self.scale_var, command=self.update_scale_and_apply)
        self.scale_slider.grid(row=1, column=7, padx=2, pady=2, sticky='ew')
        self.lbl_sca_val = tk.Label(top_frame, text="1")
        self.lbl_sca_val.grid(row=1, column=8, padx=(0,10), pady=2, sticky='w')

        # Bind Combobox selection to apply dither
        self.algo_combobox.bind("<<ComboboxSelected>>", self.apply_dither_event_wrapper)

        self.lbl_original.bind(
            '<Configure>', lambda e: self.update_image_display(
                self.lbl_original, self.original_image))
        self.lbl_dithered.bind(
            '<Configure>', lambda e: self.update_image_display(
                self.lbl_dithered, self.dithered_image))

    def update_threshold_and_apply(self, value):
        self.update_threshold_label(value)
        if self.auto_render_var.get():
            self.apply_dither()

    def update_contrast_and_apply(self, value):
        self.update_contrast_label(value)
        if self.auto_render_var.get():
            self.apply_dither()

    def update_scale_and_apply(self, value):
        self.update_scale_label(value)
        if self.auto_render_var.get():
            self.apply_dither()

    def update_rgb_or_greyscale_and_apply(self, value):
        self.update_rgb_or_greyscale_label(value)
        if self.auto_render_var.get():
            self.apply_dither()

    def apply_dither_event_wrapper(self, event=None):
        if self.auto_render_var.get():
            self.root.after(10, self.apply_dither)

    def update_threshold_label(self, value):
        val = int(float(value))
        self.lbl_thr_val.config(text=str(val))

    def update_contrast_label(self, value):
        val = f"{float(value):.1f}"
        self.lbl_con_val.config(text=val)

    def update_scale_label(self, value):
        val = int(float(value))
        self.scale_var.set(val)
        self.lbl_sca_val.config(text=str(val))

    def update_rgb_or_greyscale_label(self, value):
        # No label to update, just a checkbox toggle
        pass
    
    def floyd_steinberg_numpy(self, pil_img, threshold=128, pixel_scale=1):
        if pixel_scale <= 0:
            pixel_scale = 1

        if self.rgb_or_greyscale_var.get():
            type = 'RGB'
        else:
            type = 'L'

        if pixel_scale == 1:
            img = pil_img.convert(type)
            arr = np.array(img, dtype=np.float32)
            print("type", type)
            arr = fs_dither(arr, type, threshold)    
            
            return Image.fromarray(arr)
        else:
            orig_w, orig_h = pil_img.size
            small_w = max(1, orig_w // pixel_scale)
            small_h = max(1, orig_h // pixel_scale)

            print(f"Downscaling to {small_w}x{small_h}")
            small_img = pil_img.resize((small_w, small_h), Image.Resampling.BOX)

            print("Dithering downscaled image...")
            dithered_small_img = self.floyd_steinberg_numpy(small_img, threshold, pixel_scale=1)

            print(f"Upscaling back to {orig_w}x{orig_h}")
            dithered_large_img = dithered_small_img.resize((orig_w, orig_h), Image.Resampling.NEAREST)
            return dithered_large_img
    

    
    def simple_threshold(self, pil_img, threshold=128, pixel_scale=1):
        if pixel_scale <= 0:
            pixel_scale = 1

        if self.rgb_or_greyscale_var.get():
            type = 'RGB'
        else:
            type = 'L'
        img_gray = pil_img.convert(type)

        if pixel_scale == 1:
            if type == 'RGB':
                # Process each band separately for RGB using NumPy
                img_array = np.array(img_gray)
                result = simple_threshold_rgb_ps1(img_array, threshold)
                return Image.fromarray(result.astype(np.uint8))
            else:
                # For grayscale, use PIL's point for threshold (faster for single channel)
                return img_gray.point(lambda p: 0 if p < threshold else 255)
        else:
            # For larger pixel scales, we need the block-based approach
            orig_w, orig_h = img_gray.size
            
            # Convert to numpy array
            img_array = np.array(img_gray)
            
            # Create output array of same shape
            out_array = simple_threshold_dither(img_array, type, pixel_scale, orig_w, orig_h, threshold)
            
            return Image.fromarray(out_array)

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        if not path:
            return
        self.open_path = path
        try:
            self.original_image = Image.open(path)
        except Exception as e:
            print(f"Error opening image: {e}")
            self.original_image = None
            self.dithered_image = None
            self.update_image_display(self.lbl_original, None)
            self.update_image_display(self.lbl_dithered, None)
            self.save_button.config(state=tk.DISABLED)
            return # Stop further processing on error
            
        # Reset dithered image display and state
        self.dithered_image = None
        self.update_image_display(self.lbl_original, self.original_image) 
        self.lbl_dithered.config(image='') # Clear dithered display explicitly
        self.lbl_dithered.image = None # Clear reference for dithered display
        
        # Apply dither immediately if auto-render is enabled
        if self.auto_render_var.get():
            print("Auto-rendering after image open.") # Debug print
            self.apply_dither()
            
        # Enable save button regardless of auto-render state if image loaded
        self.save_button.config(state=tk.NORMAL)

    def apply_dither(self):
        if not self.original_image:
            return

        alg = self.algorithm_var.get()
        thr = int(self.threshold_slider.get())
        contrast_factor = self.contrast_var.get()
        pixel_s = self.scale_var.get()

        print(f"Applying {alg} with threshold {thr}, contrast {contrast_factor:.2f}, scale {pixel_s}")

        image_to_dither = self.original_image
        if abs(contrast_factor - 1.0) > 0.01:
            try:
                enhancer = ImageEnhance.Contrast(self.original_image)
                image_to_dither = enhancer.enhance(contrast_factor)
                print("Applied contrast adjustment.")
            except Exception as e:
                print(f"Error applying contrast: {e}")
                image_to_dither = self.original_image

        if alg == "Floyd-Steinberg":
            self.dithered_image = self.floyd_steinberg_numpy(image_to_dither, thr, pixel_s)
        elif alg == "Simple Threshold":
            self.dithered_image = self.simple_threshold(image_to_dither, thr, pixel_s)
        else:
            self.dithered_image = None

        self.update_image_display(self.lbl_dithered, self.dithered_image)

    def update_image_display(self, label, pil_image):
        w = label.winfo_width()
        h = label.winfo_height()
        if not pil_image or w < 5 or h < 5:
            return
        img_copy = pil_image.copy()
        img_copy.thumbnail((w, h), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(img_copy)
        label.config(image=tk_img)
        label.image = tk_img

    def save_image(self):
        if not self.dithered_image:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All Files", "*.*")]
        )
        if path:
            self.dithered_image.save(path)

if __name__ == "__main__":
    root = tk.Tk()
    app = DitherApp(root)
root.mainloop()
