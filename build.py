import platform
import subprocess
import os
import shutil
import config
def run_pyinstaller(command):
    """Runs a PyInstaller command and prints output."""
    print(f"Running PyInstaller: {' '.join(command)}")
    try:
        # Using shell=True might be needed on Windows if PATH issues occur,
        # but passing a list is generally safer if possible.
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        print("PyInstaller output:")
        print(process.stdout)
        print("Build done, cleaning up...")

        # Remove spec file after build
        if config.REMOVE_SPEC:
            spec_file = f"{config.APP_NAME}.spec"
            if os.path.isfile(spec_file):
                print(f"Removing spec file: {spec_file}")
                os.remove(spec_file)

        print("\nBuild successful!")
        print(f"Build complete, enjoy {config.APP_NAME}!")
    except subprocess.CalledProcessError as e:
        print("!!! PyInstaller failed !!!")
        print(f"Return code: {e.returncode}")
        print("--- stdout ---")
        print(e.stdout)
        print("--- stderr ---")
        print(e.stderr)
    except FileNotFoundError:
        print(f"Error: '{command[0]}' command not found.")
        print("Make sure PyInstaller is installed and in your PATH.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def clean_build():
    """Removes old build artifacts."""
    
    print("Cleaning previous build artifacts...")
    for folder in ["build", "dist"]:
        if os.path.isdir(folder):
            print(f"Removing folder: {folder}")
            shutil.rmtree(folder)

    # Remove spec file if it exists
    spec_file = f"{os.path.splitext(config.SCRIPT_NAME)[0]}.spec"
    if os.path.isfile(spec_file):
            print(f"Removing file: {spec_file}")
            os.remove(spec_file)

# And now, le code...

# Determine the current OS
current_os = platform.system()
print(f"Detected OS: {current_os}")

# Clean previous build
clean_build()


# Add OS-specific icon
if current_os == "Windows" or current_os == "Darwin":
    if os.path.exists(config.ICON):
        config.pyinstaller_cmd.extend(["--icon", config.ICON])
    else:
        print(f"Warning: icon '{config.ICON}' not found. Building without icon.")
else: # Linux or other
    print("Note: Linux icons are typically handled via .desktop files, not directly in the executable.")
    # No specific icon flag added for generic Linux builds via PyInstaller

# Add the main script
config.pyinstaller_cmd.append(config.SCRIPT_NAME)

# Run the build command
run_pyinstaller(config.pyinstaller_cmd)