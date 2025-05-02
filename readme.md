# Interactive Dither App

A simple desktop application built with Python and Tkinter/Pillow to apply Floyd-Steinberg dithering to images interactively.

## What it does

*   Open common image file formats (PNG, JPG, BMP, GIF, TIFF).
*   Display the original and dithered images side-by-side in a scrollable view.
*   Apply Floyd-Steinberg dithering.
*   Adjust the grayscale threshold used for dithering with a slider.
*   Manually apply the dithering effect using a button after adjusting the threshold.
*   Save the resulting dithered image, allowing you to choose the filename and location.
*   Image previews dynamically resize to fit the application window.

## Setup

1.  **Clone the repository (or download the code):**
    ```bash
    # If using git
    git clone <repository-url>
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

4.  **Install dependencies:** The main dependencies are Pillow (for image manipulation), NumPy (for faster array processing), and TKinter for the UI. The project also uses subprocess and os libraries.
    ```bash
    pip install Pillow numpy
    ```
    *(Note: Tkinter (ttk) is usually included with Python standard distributions. If you encounter `_tkinter` errors, especially on macOS, ensure your Python installation includes Tk support.)*

## Usage

Once the setup is complete and your virtual environment is active, run the main script:

```bash
python main.py
```

This will launch the application window. Use the "Open Image" button to load an image, adjust the "Threshold" slider, click "Apply Dither" to see the result, and use "Save Dithered Image" to export the output.

## Implementation Notes

*   **NumPy for Floyd-Steinberg:** The Floyd-Steinberg algorithm (`floyd_steinberg_numpy` function) is implemented using NumPy. This was done to significantly speed up the pixel processing compared to iterating through pixels using standard PIL methods (`Image.load()`). The image is converted to a NumPy array, the dithering calculations and error diffusion are performed directly on the array elements, and the resulting array is converted back to a PIL Image. This array-based approach is generally much faster for image manipulation tasks.
*   **Image Resizing:** Image previews in the GUI are dynamically resized using PIL's `thumbnail` method to fit the allocated space in the labels. The actual dithering and saving operations are performed on the original image data.

## Motivation & Future Thoughts

> I started this project after seeing someone sell a dithering software for 35. Its very fucking slow btw I'm thinking of implementing it another way so that maybe I could use it with live images.
