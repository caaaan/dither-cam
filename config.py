#A config file to serve variables to both build.py and main.py files

# --- Configuration ---
APP_NAME = "Dither3000"
SCRIPT_NAME = "main.py"
ICON = "ditherer.ico" # Put your custom icon in the root directory 
REMOVE_SPEC = True # Remove the spec file after build

# Camera configuration
CAMERA_SAFE_INIT = True  # Set to False to skip extensive camera initialization checks

pyinstaller_cmd = [      # Base PyInstaller command
    "pyinstaller",
    "--windowed",        # For GUI apps (no console)
    "--name", APP_NAME,  # Set the application name
    "--onefile",       # Uncomment if you prefer a single file build
]
# --- End Configuration ---
