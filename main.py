import sys
from PyQt6.QtWidgets import QApplication
from classes.dither_app import DitherApp

if __name__ == "__main__":
    # Create the Qt Application
    app = QApplication(sys.argv)
    
    # Create and show the main application window
    window = DitherApp()
    
    # Start the application main loop
    sys.exit(app.exec())
