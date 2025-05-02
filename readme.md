##### Interactive dither app or as I'd like to call it,

<div align="center">
  <p style="font-size: 2em; font-weight: bold; margin-bottom: 4; font-family: 'Courier New', monospace;">Dither3000</p>
 
<img src="ditherer.jpeg" alt="logo" width="100" style="margin-bottom: 4; "/>
</div>

A simple desktop application built with Python and Tkinter/Pillow to apply dithering algorithms to images interactively, with controls for threshold, contrast, and pixel scale.

## What it does

*   Open common image file formats (PNG, JPG, BMP, GIF, TIFF).
*   Display the original and dithered images side-by-side.
*   Apply Floyd-Steinberg (error diffusion) or Simple Threshold dithering.
*   Adjust the grayscale threshold used for dithering via a slider.
*   Adjust the image contrast *before* dithering via a slider.
*   Adjust the effective "pixel scale" for dithering (1 = normal, >1 = blocky effect) via a slider.
*   Optionally enable "Auto-Render" for live updates as sliders change.
*   Manually apply the dithering effect using a button.
*   Save the resulting dithered image, allowing you to choose the filename and location.
*   Image previews dynamically resize to fit the application window.

## Example Screenshot

![Example Screenshot](example.png)



## Setup

1.  **Clone the repository (or download the code):**
    ```bash
    # If using git
    git clone https://github.com/caaaan/basic-dither.git # Replace with your repo URL if applicable
    cd dither-app
    ```

2.  **Ensure you have Python 3 installed.** You can check with `python3 --version`.

3.  **Create and activate a virtual environment (recommended):**
    ```bash
    # macOS / Linux
    python3 -m venv .venv
    source .venv/bin/activate

    # Windows (cmd.exe)
    python -m venv .venv
    .venv\Scripts\activate.bat

    # Windows (PowerShell)
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    ```

4.  **Install dependencies:** The main dependencies are Pillow (for image manipulation) and NumPy (for faster array processing).
    ```bash
    # Using pip (recommended within virtual environment)
    pip install Pillow numpy

    # Or using Conda (if you prefer conda environments)
    # conda install pillow numpy
    ```
    *(Note: Tkinter (ttk) is usually included with Python standard distributions. If you encounter `_tkinter` errors, especially on macOS, ensure your Python installation includes Tk support.)*

## Usage

Once the setup is complete and your virtual environment is active, run the main script:

```bash
python main.py
```

This will launch the application window.
1.  Use the "Open Image" button to load an image.
2.  Select the desired "Algorithm".
3.  Adjust the "Threshold", "Contrast", and "Pixel Scale" sliders.
4.  If "Auto-Render" is checked, the preview updates automatically. If unchecked, click "Apply Dither" to see the result.
5.  Use "Save Image" to export the output.

## Implementation Notes

*   **NumPy for Floyd-Steinberg:** The Floyd-Steinberg algorithm (`floyd_steinberg_numpy` function) is implemented using NumPy for significant speed improvements over standard PIL pixel iteration. It converts the image to a NumPy array, performs calculations directly on the array, and converts back.
*   **Contrast Adjustment:** The contrast slider uses `PIL.ImageEnhance.Contrast` to modify the image *before* it's passed to the selected dithering algorithm.
*   **Pixel Scale Implementation:**
    *   For *Simple Threshold*, a scale > 1 causes the algorithm to process the image in `scale x scale` blocks, calculating the average grayscale value for each block, thresholding that average, and filling the output block with the result.
    *   For *Floyd-Steinberg*, a scale > 1 uses a downscale-dither-upscale approach: the image is first downscaled by the scale factor (using `Image.Resampling.BOX` averaging), the standard dithering algorithm is applied to the small image, and the result is upscaled back to the original size (using `Image.Resampling.NEAREST` to preserve blocks).
*   **Image Resizing:** Image previews in the GUI are dynamically resized using PIL's `thumbnail` method to fit the allocated space in the labels. The actual dithering and saving operations are performed on the contrast-adjusted image data at the selected effective scale.

## Building the Application

To create a standalone executable (.exe on Windows, .app on macOS) that can be run without installing Python or dependencies, you can use the included `build_app.py` script.

1.  **Prerequisites:**
    *   Install PyInstaller: `pip install pyinstaller`
    *   Create icon files named `ditherer.ico` (for Windows) and `ditherer.icns` (for macOS) from an image and place them in the same directory as the script.
2.  **Run the build script:**
    ```bash
    python build_app.py
    ```
This script will automatically detect your operating system, clean previous builds, and run PyInstaller with the appropriate settings (including the correct icon file) to create the application bundle in the `dist` folder. You must run this script on the target OS (run on Windows to build for Windows, run on macOS to build for macOS).

## Motivation & Future Thoughts

> I started this project after seeing someone sell a dithering software for 35. Its very fucking slow btw I'm thinking of implementing it another way so that maybe I could use it with live images.
