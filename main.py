import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import subprocess
import os

class DitherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Ditherer")
        self.root.geometry("800x600")
        self.original_image = None
        self.dithered_image = None

        # --- Layout ---
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        top_frame = tk.Frame(root)
        top_frame.grid(row=0, column=0, sticky='ew', padx=10, pady=5)
        top_frame.grid_columnconfigure(7, weight=1)

        btn_open = tk.Button(
            top_frame, text="Open Image", command=self.open_image,
            bg='lightgray', activebackground='blue', activeforeground='white'
        )
        btn_open.grid(row=0, column=0, padx=5)

        btn_save = tk.Button(
            top_frame, text="Save Dithered Image", command=self.save_image,
            state=tk.DISABLED, bg='lightgray', activebackground='blue', activeforeground='white'
        )
        btn_save.grid(row=0, column=1, padx=5)
        self.save_button = btn_save

        lbl_algo = tk.Label(top_frame, text="Algorithm:")
        lbl_algo.grid(row=0, column=2, padx=(10,5))
        self.algorithm_var = tk.StringVar()
        self.algo_combobox = ttk.Combobox(
            top_frame, textvariable=self.algorithm_var,
            values=["Floyd-Steinberg", "Simple Threshold"], state='readonly'
        )
        self.algo_combobox.current(0)
        self.algo_combobox.grid(row=0, column=3, padx=5)

        lbl_threshold = tk.Label(top_frame, text="Threshold:")
        lbl_threshold.grid(row=0, column=4, padx=(10,5))
        self.threshold_slider = ttk.Scale(
            top_frame, from_=1, to=254, orient=tk.HORIZONTAL,
            command=self.update_threshold_label
        )
        self.threshold_slider.set(128)
        self.threshold_slider.grid(row=0, column=5, padx=5)

        self.lbl_thr_val = tk.Label(top_frame, text="128")
        self.lbl_thr_val.grid(row=0, column=6, padx=(0,10))

        btn_apply = tk.Button(
            top_frame, text="Apply Dither", command=self.apply_dither,
            bg='lightgray', activebackground='blue', activeforeground='white'
        )
        btn_apply.grid(row=0, column=7, padx=5)

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

        self.lbl_original.bind(
            '<Configure>', lambda e: self.update_image_display(
                self.lbl_original, self.original_image))
        self.lbl_dithered.bind(
            '<Configure>', lambda e: self.update_image_display(
                self.lbl_dithered, self.dithered_image))

    def update_threshold_label(self, value):
        val = int(float(value))
        self.lbl_thr_val.config(text=str(val))

    def floyd_steinberg_numpy(self, pil_img, threshold=128):
        img = pil_img.convert('L')
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape
        for y in range(h):
            for x in range(w):
                old = arr[y, x]
                new = 0 if old < threshold else 255
                err = old - new
                arr[y, x] = new
                if x+1 < w:
                    arr[y, x+1] += err * 7/16
                if y+1 < h:
                    if x > 0:
                        arr[y+1, x-1] += err * 3/16
                    arr[y+1, x] += err * 5/16
                    if x+1 < w:
                        arr[y+1, x+1] += err * 1/16
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def simple_threshold(self, pil_img, threshold=128):
        img = pil_img.convert('L')
        return img.point(lambda p: 0 if p < threshold else 255)

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        if not path:
            return
        self.open_path = path
        self.original_image = Image.open(path)
        self.dithered_image = None
        self.update_image_display(self.lbl_original, self.original_image)
        self.save_button.config(state=tk.NORMAL)
        self.lbl_dithered.config(image='')

    def apply_dither(self):
        if not self.original_image:
            return
        alg = self.algorithm_var.get()
        thr = int(self.threshold_slider.get())
        print(f"Applying {alg} with threshold {thr}")

        if alg == "Floyd-Steinberg":
            self.dithered_image = self.floyd_steinberg_numpy(self.original_image, thr)
        elif alg == "Simple Threshold":
            self.dithered_image = self.simple_threshold(self.original_image, thr)
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
