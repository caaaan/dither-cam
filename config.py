#A config file to serve variables to both build.py and main.py files

# --- Configuration ---

#Build configuration
APP_NAME = "Dithercam"
SCRIPT_NAME = "main.py"
ICON = "ditherer.ico" # Put your custom icon in the root directory 
REMOVE_SPEC = True # Remove the spec file after build

# Camera configuration
CAMERA_SAFE_INIT = False  # Set to False to skip extensive camera initialization checks

# Image capture configuration
DEFAULT_CAPTURE_FORMAT = "png"  # File extension for captured images (png, jpg, jpeg)

# GPIO pin configuration (BCM numbering)
GPIO_UP = 16
GPIO_DOWN = 26
GPIO_LEFT = 20
GPIO_RIGHT = 19
GPIO_CAPTURE = 21

pyinstaller_cmd = [      # Base PyInstaller command
    "pyinstaller",
    "--windowed",        # For GUI apps (no console)
    "--name", APP_NAME,  # Set the application name
    "--onefile",       # Uncomment if you prefer a single file build
]
# --- End Configuration ---
