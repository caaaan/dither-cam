import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                             QLabel, QVBoxLayout, QHBoxLayout, QComboBox, 
                             QSlider, QFileDialog, QSplitter, QCheckBox,
                             QFrame, QScrollArea)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QWheelEvent
from PIL import Image, ImageEnhance
import numpy as np
import os, config
from helper import fs_dither, simple_threshold_rgb_ps1, simple_threshold_dither

class ImageViewer(QScrollArea):
    zoom_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: transparent; border: 1px solid #ddd;")
        
        self.setWidget(self.image_label)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        self.original_pixmap = None
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0

    def _apply_zoom_to_display(self):
        """Scales original_pixmap by zoom_factor and updates the label."""
        if not self.original_pixmap:
            self.image_label.clear()
            return

        new_width = int(self.original_pixmap.width() * self.zoom_factor)
        new_height = int(self.original_pixmap.height() * self.zoom_factor)
        
        scaled_pixmap = self.original_pixmap.scaled(
            new_width, new_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(scaled_pixmap.size())
        
    def set_image(self, pixmap):
        if pixmap:
            self.original_pixmap = pixmap.copy()
            # Apply the current internal zoom_factor to the new pixmap.
            # Do not change self.zoom_factor here.
            self._apply_zoom_to_display()
        else: # Image is being cleared
            self.original_pixmap = None
            self._apply_zoom_to_display() # This will clear the label
            # If image is cleared, reset internal zoom and notify to reset slider.
            if abs(self.zoom_factor - 1.0) > 0.001:
                self.zoom_factor = 1.0
                self.zoom_changed.emit(1.0)
            elif self.image_label.pixmap() is None: # Ensure emit if already 1.0 but no image
                 self.zoom_changed.emit(1.0)

    def set_zoom_level(self, level: float):
        new_zoom_factor = max(self.min_zoom, min(self.max_zoom, level))

        # Only update if the factor actually changes or if there's no image (to set initial zoom state)
        if abs(self.zoom_factor - new_zoom_factor) > 0.001 or (not self.original_pixmap and self.image_label.pixmap() is None):
            self.zoom_factor = new_zoom_factor
            self._apply_zoom_to_display()
            self.zoom_changed.emit(self.zoom_factor)
            
    def wheelEvent(self, event: QWheelEvent):
        # if not self.original_pixmap: # Covered by set_zoom_level not applying if no pixmap
        #     super().wheelEvent(event)
        #     return

        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            angle_delta = event.angleDelta().y()
            proposed_zoom_factor = self.zoom_factor
            
            if angle_delta > 0:
                proposed_zoom_factor *= 1.1
            else:
                proposed_zoom_factor *= 0.9
            
            self.set_zoom_level(proposed_zoom_factor)
            event.accept()
        else:
            super().wheelEvent(event)

class DitherApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.APP_NAME)
        self.resize(1200, 800)
        
        self.original_image = None
        self.dithered_image = None
        self.showing_original = False # Show dithered version first by default
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create main splitter widget
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(self.splitter)
        
        # Left panel (80%) - Image display
        self.image_panel = QWidget()
        self.image_layout = QVBoxLayout(self.image_panel)
        
        # Image viewer with scrollbars
        self.image_viewer = ImageViewer()
        self.image_layout.addWidget(self.image_viewer, 1)
        
        # "Switch to Original Image" button - will be created and styled later for control_panel
        self.toggle_button = QPushButton("Switch to Original Image")
        self.toggle_button.clicked.connect(self.toggle_image_display)
        self.toggle_button.setEnabled(False)
        self.toggle_button.setMinimumWidth(200)
        self.toggle_button.setFixedHeight(40)
        
        # Note: toggle_button is NOT added to image_layout here anymore
        
        # Right panel (20%) - Controls
        self.control_panel = QWidget()
        self.control_panel.setObjectName("controlPanelBackground")

        # Assuming 'control_panel_background.png' is in the same directory as main.py
       
        # More specific stylesheet for control_panel and general styles for its children
        self.control_panel.setStyleSheet(f"""
            #controlPanelBackground {{
                border-image: url({os.path.join(os.path.dirname(__file__), "background.jpg")}) 0 0 0 0 stretch stretch;
            }}

            /* Styles for children of controlPanelBackground for readability */
            #controlPanelBackground QLabel {{
                background-color: transparent;
                color: white;
            }}
            #controlPanelBackground QCheckBox {{
                background-color: transparent;
                color: white;
            }}
            #controlPanelBackground QPushButton {{
                border: 1px solid #BBB;
                border-radius: 15px; 
                padding: 5px;
                background-color: rgba(200, 200, 200, 0.3); /* Semi-transparent background */
                color: white;
            }}
            #controlPanelBackground QPushButton:hover {{
                background-color: rgba(220, 220, 220, 0.5);
            }}
            #controlPanelBackground QPushButton:pressed {{
                background-color: rgba(180, 180, 180, 0.5);
            }}
        """)

        self.control_layout = QVBoxLayout(self.control_panel)
        self.control_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.control_layout.setContentsMargins(10, 20, 10, 20)
        self.control_layout.setSpacing(10)
        
        control_title = QLabel("Controls")
        # control_title style will be inherited or can be set specifically if needed
        self.control_layout.addWidget(control_title)
        
        button_layout = QHBoxLayout()
        self.open_button = QPushButton("Open Image")
        self.open_button.clicked.connect(self.open_image)
        self.open_button.setMinimumWidth(120) # Style will come from panel stylesheet
        button_layout.addWidget(self.open_button)
        
        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)
        self.save_button.setMinimumWidth(120) # Style will come from panel stylesheet
        button_layout.addWidget(self.save_button)
        
        button_container = QWidget() # This container itself won't get the background image
        button_container.setLayout(button_layout)
        self.control_layout.addWidget(button_container, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Add horizontal divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        self.control_layout.addWidget(divider)
        
        # Zoom Slider
        self.zoom_label = QLabel("Zoom: 100%")
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.control_layout.addWidget(self.zoom_label)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 500)  # Represents 10% to 500%
        self.zoom_slider.setValue(100)       # Default 100% (1.0x zoom)
        self.zoom_slider.setTickInterval(10)
        self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.zoom_slider.valueChanged.connect(self.handle_zoom_slider_change)
        self.zoom_slider.setMinimumWidth(200)
        self.control_layout.addWidget(self.zoom_slider, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Algorithm selector 
        self.algorithm_label = QLabel("Algorithm:")
        self.algorithm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.control_layout.addWidget(self.algorithm_label)
        
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["Floyd-Steinberg", "Simple Threshold"])
        self.algorithm_combo.currentIndexChanged.connect(self.algorithm_changed)
        self.algorithm_combo.setMinimumWidth(180)
        self.control_layout.addWidget(self.algorithm_combo, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Threshold slider
        self.threshold_label = QLabel("Threshold: 128")
        self.threshold_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.control_layout.addWidget(self.threshold_label)
        
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(1, 254)
        self.threshold_slider.setValue(128)
        self.threshold_slider.valueChanged.connect(self.threshold_changed)
        self.threshold_slider.setMinimumWidth(200)
        self.control_layout.addWidget(self.threshold_slider, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Contrast slider
        self.contrast_label = QLabel("Contrast: 1.0")
        self.contrast_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.control_layout.addWidget(self.contrast_label)
        
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(10, 500)  # 0.1 to 5.0 scaled by 100
        self.contrast_slider.setValue(100)      # 1.0 * 100
        self.contrast_slider.valueChanged.connect(self.contrast_changed)
        self.contrast_slider.setMinimumWidth(200)
        self.control_layout.addWidget(self.contrast_slider, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Pixel Scale slider
        self.scale_label = QLabel("Pixel Scale: 1")
        self.scale_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.control_layout.addWidget(self.scale_label)
        
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setRange(1, 8)
        self.scale_slider.setValue(1)
        self.scale_slider.valueChanged.connect(self.scale_changed)
        self.scale_slider.setMinimumWidth(200)
        self.control_layout.addWidget(self.scale_slider, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Add another divider
        divider2 = QFrame()
        divider2.setFrameShape(QFrame.Shape.HLine)
        divider2.setFrameShadow(QFrame.Shadow.Sunken)
        self.control_layout.addWidget(divider2)
        
        # Auto-render and RGB checkboxes
        checkbox_container = QWidget()
        checkbox_layout = QHBoxLayout(checkbox_container)
        
        self.auto_render = QCheckBox("Auto-Render")
        self.auto_render.setChecked(True)
        checkbox_layout.addWidget(self.auto_render)
        
        self.rgb_mode = QCheckBox("Color")
        self.rgb_mode.setChecked(True)
        self.rgb_mode.stateChanged.connect(self.rgb_changed)
        checkbox_layout.addWidget(self.rgb_mode)
        
        self.control_layout.addWidget(checkbox_container, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Create Apply Dither button (will be added to its own group later)
        self.apply_button = QPushButton("Apply Dither")
        self.apply_button.clicked.connect(self.apply_dither)
        self.apply_button.setMinimumWidth(200)
        self.apply_button.setFixedHeight(40)
        # Individual style removed, will be handled by control_panel stylesheet

        # Add a stretch to push the following button group towards the bottom
        self.control_layout.addStretch(1)

        # Vertical button group for Toggle and Apply buttons
        action_button_group_widget = QWidget()
        action_button_group_layout = QVBoxLayout(action_button_group_widget)
        action_button_group_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        action_button_group_layout.setSpacing(10) # Spacing between toggle and apply
        
        action_button_group_layout.addWidget(self.toggle_button) # Switch button on top
        action_button_group_layout.addWidget(self.apply_button)  # Apply button below
        
        self.control_layout.addWidget(action_button_group_widget, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Final stretch to take up any remaining space at the very bottom
        self.control_layout.addStretch(0) 
        
        # Add both panels to splitter with 80/20 ratio
        self.splitter.addWidget(self.image_panel)
        self.splitter.addWidget(self.control_panel)
        
        # Calculate the actual 80/20 split based on the window size
        total_width = self.width()
        self.splitter.setSizes([int(total_width * 0.8), int(total_width * 0.2)])
        
        # Connect ImageViewer zoom changes to update slider/label
        self.image_viewer.zoom_changed.connect(self.update_controls_from_zoom_factor)
    
    def toggle_image_display(self):
        if not self.original_image or not self.dithered_image:
            return
        
        self.showing_original = not self.showing_original
        
        if self.showing_original:
            self.display_image(self.original_image)
            self.toggle_button.setText("Switch to Original Image")
        else:
            self.display_image(self.dithered_image)
            self.toggle_button.setText("Switch to Original Image")
    
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if not path:
            return
        
        self.open_path = path
        try:
            self.original_image = Image.open(path)
            # self.showing_original is already False from __init__
            
            self.apply_dither() # Attempt to dither and display immediately
            self.image_viewer.set_zoom_level(1.0) # Reset zoom for the new image view

            if self.dithered_image:
                # Dithering was successful, dithered image is shown
                self.toggle_button.setText("Switch to Original Image")
                self.save_button.setEnabled(True)
                self.toggle_button.setEnabled(True) # Original also exists
            else:
                # Initial dithering failed, fall back to showing original
                self.showing_original = True # Update state to reflect original is shown
                self.display_image(self.original_image) # Explicitly display original
                self.toggle_button.setText("Switch to Original Image")
                self.save_button.setEnabled(False)
                self.toggle_button.setEnabled(False) # No dithered image to switch to

            # The auto_render check here is no longer needed as we try to dither by default.
            # if self.auto_render.isChecked():
            #     self.apply_dither()
                
        except Exception as e:
            print(f"Error opening image: {e}")
            self.original_image = None
            self.dithered_image = None
            self.image_viewer.set_image(None) 
            self.save_button.setEnabled(False)
            self.toggle_button.setEnabled(False)
            self.showing_original = False # Reset for next attempt
    
    def display_image(self, pil_image):
        if not pil_image:
            self.image_viewer.set_image(None) # set_image(None) handles zoom reset
            return
        
        img_copy = pil_image.copy()
        
        # Convert PIL image to QPixmap at original size
        if img_copy.mode == "RGBA":
            img_data = img_copy.tobytes()
            q_image = QImage(
                img_data, 
                img_copy.width, 
                img_copy.height, 
                img_copy.width * 4, # RGBA has 4 bytes per pixel
                QImage.Format.Format_RGBA8888
            )
        elif img_copy.mode == "RGB":
            img_data = img_copy.tobytes()
            q_image = QImage(
                img_data, 
                img_copy.width, 
                img_copy.height, 
                img_copy.width * 3, 
                QImage.Format.Format_RGB888
            )
        elif img_copy.mode == "L":
            img_data = img_copy.tobytes()
            q_image = QImage(
                img_data, 
                img_copy.width, 
                img_copy.height, 
                img_copy.width, 
                QImage.Format.Format_Grayscale8
            )
        elif img_copy.mode == "P": # Palette-based images
            img_copy = img_copy.convert("RGBA") # Convert to RGBA to handle transparency properly
            img_data = img_copy.tobytes()
            q_image = QImage(
                img_data,
                img_copy.width,
                img_copy.height,
                img_copy.width * 4,
                QImage.Format.Format_RGBA8888
            )
        else: # For other modes, try converting to RGBA
            try:
                img_copy = img_copy.convert("RGBA")
                img_data = img_copy.tobytes()
                q_image = QImage(
                    img_data, 
                    img_copy.width, 
                    img_copy.height, 
                    img_copy.width * 4, 
                    QImage.Format.Format_RGBA8888
                )
            except Exception as e:
                print(f"Could not convert image mode {img_copy.mode} to RGBA: {e}")
                self.image_viewer.set_image(None)
                return
        
        pixmap = QPixmap.fromImage(q_image)
        self.image_viewer.set_image(pixmap)
    
    def threshold_changed(self, value):
        self.threshold_label.setText(f"Threshold: {value}")
        if self.auto_render.isChecked():
            self.apply_dither()
    
    def contrast_changed(self, value):
        contrast = value / 100.0
        self.contrast_label.setText(f"Contrast: {contrast:.1f}")
        if self.auto_render.isChecked():
            self.apply_dither()
    
    def scale_changed(self, value):
        self.scale_label.setText(f"Pixel Scale: {value}")
        if self.auto_render.isChecked():
            self.apply_dither()
    
    def algorithm_changed(self):
        if self.auto_render.isChecked():
            self.apply_dither()
    
    def rgb_changed(self):
        if self.auto_render.isChecked():
            self.apply_dither()
    
    def apply_dither(self):
        if not self.original_image:
            return
        
        alg = self.algorithm_combo.currentText()
        thr = self.threshold_slider.value()
        contrast_factor = self.contrast_slider.value() / 100.0
        pixel_s = self.scale_slider.value()
        
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
        
        # Update toggle button state
        if self.dithered_image:
            self.toggle_button.setEnabled(True)
            
            # If not showing original, update display with new dithered image
            if not self.showing_original:
                self.display_image(self.dithered_image)
    
    def save_image(self):
        if not self.dithered_image:
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "PNG (*.png);;JPEG (*.jpg);;All Files (*.*)"
        )
        if path:
            self.dithered_image.save(path)
    
    def floyd_steinberg_numpy(self, pil_img, threshold=128, pixel_scale=1):
        if pixel_scale <= 0:
            pixel_scale = 1
        
        if self.rgb_mode.isChecked():
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
        
        if self.rgb_mode.isChecked():
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

    def handle_zoom_slider_change(self, value):
        zoom_level = value / 100.0  # Convert slider value (10-500) to factor (0.1-5.0)
        # Update label immediately for responsiveness, even if viewer clamps it
        self.zoom_label.setText(f"Zoom: {value}%") 
        self.image_viewer.set_zoom_level(zoom_level)

    def update_controls_from_zoom_factor(self, zoom_factor):
        slider_value = int(zoom_factor * 100)
        # Block signals from slider to prevent feedback loop if slider.setValue itself triggers valueChanged
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(slider_value)
        self.zoom_slider.blockSignals(False)
        self.zoom_label.setText(f"Zoom: {slider_value}%")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Update splitter ratio when window is resized
        total_width = self.width()
        self.splitter.setSizes([int(total_width * 0.8), int(total_width * 0.2)])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DitherApp()
    window.show()
    sys.exit(app.exec())
