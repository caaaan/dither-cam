import time
import threading
from gpiozero import Button

class ButtonConnect:
    """
    Interface for GPIO button connections on Raspberry Pi using gpiozero.
    Handles button press detection and provides callbacks for button events.
    """
    
    # Button states
    BUTTON_RELEASED = 0
    BUTTON_PRESSED = 1
    
    def __init__(self, button_pins=None, bounce_time=50, pull_up=True):
        """
        Initialize GPIO and set up button pins.
        
        Args:
            button_pins (dict): Dictionary mapping button names to GPIO pin numbers
                                Example: {'up': 17, 'down': 18, 'left': 22, 'right': 23, 'select': 27}
            bounce_time (int): Debounce time in milliseconds
            pull_up (bool): True to use internal pull-up resistors (button connects to GND),
                           False to use pull-down (button connects to 3.3V)
        """
        # Default button pins if none provided
        self.button_pins = button_pins or {
            'up': 16,
            'down': 26,
            'left': 20,
            'right': 19,
            'select': 21
        }
        
        # Button state tracking
        self.button_states = {button: self.BUTTON_RELEASED for button in self.button_pins}
        self.last_press_time = {button: 0 for button in self.button_pins}
        self.callbacks = {}
        
        # Setup parameters
        self.bounce_time_s = bounce_time / 1000.0  # Convert ms to seconds for gpiozero
        self.pull_up = pull_up
        
        # Create button objects
        self.buttons = {}
        
        print(f"Initializing buttons with pull_up={pull_up}")
        for button_name, pin in self.button_pins.items():
            print(f"Setting up {button_name} button on GPIO {pin}")
            try:
                # Create gpiozero Button object with correct parameters
                btn = Button(pin=pin, pull_up=pull_up, bounce_time=self.bounce_time_s)
                
                # Store internal reference to this button
                self.buttons[button_name] = btn
                
                # Define proper callbacks using lambda with default args to capture current value
                btn.when_pressed = lambda b=button_name: self._handle_button_pressed(b)
                btn.when_released = lambda b=button_name: self._handle_button_released(b)
                
                print(f"  {button_name} initialized on GPIO {pin}")
            except Exception as e:
                print(f"Error initializing button {button_name} on pin {pin}: {e}")
                
        # Polling thread for continuous state monitoring
        self.polling = False
        self.poll_thread = None
    
    def start_polling(self, interval=0.05):
        """
        Start a polling thread to continuously monitor button states.
        
        Args:
            interval (float): Polling interval in seconds
        """
        if self.polling:
            return
            
        self.polling = True
        self.poll_thread = threading.Thread(target=self._poll_buttons, args=(interval,))
        self.poll_thread.daemon = True
        self.poll_thread.start()
        print("Button polling started")
    
    def stop_polling(self):
        """Stop the polling thread"""
        self.polling = False
        if self.poll_thread:
            self.poll_thread.join(timeout=1.0)
            self.poll_thread = None
    
    def _poll_buttons(self, interval):
        """
        Continuously poll button states.
        
        Args:
            interval (float): Polling interval in seconds
        """
        while self.polling:
            for button_name in self.button_pins.keys():
                if button_name in self.buttons:
                    # Update button state from gpiozero object
                    is_pressed = self.buttons[button_name].is_pressed
                    self.button_states[button_name] = self.BUTTON_PRESSED if is_pressed else self.BUTTON_RELEASED
            time.sleep(interval)
    
    def _handle_button_pressed(self, button):
        """
        Handle button press events and invoke callbacks.
        
        Args:
            button (str): Button name
        """
        # Update button press time
        current_time = time.time()
        self.last_press_time[button] = current_time
        
        # Update state
        self.button_states[button] = self.BUTTON_PRESSED
        print(f"Button pressed: {button}")
        
        # Call button-specific callback if registered
        if button in self.callbacks:
            self.callbacks[button](button)
        
        # Call general callback if registered
        if 'all' in self.callbacks:
            self.callbacks['all'](button)
    
    def _handle_button_released(self, button):
        """
        Handle button release events.
        
        Args:
            button (str): Button name
        """
        self.button_states[button] = self.BUTTON_RELEASED
        print(f"Button released: {button}")
    
    def register_callback(self, button, callback_function):
        """
        Register a callback function for a specific button.
        
        Args:
            button (str): Button name or 'all' for all buttons
            callback_function (callable): Function to call when button is pressed
        """
        self.callbacks[button] = callback_function
        print(f"Registered callback for button: {button}")
    
    def is_pressed(self, button):
        """
        Check if a button is currently pressed.
        
        Args:
            button (str): Button name
        
        Returns:
            bool: True if the button is pressed, False otherwise
        """
        if button not in self.buttons:
            return False
        
        return self.buttons[button].is_pressed
    
    def was_pressed_recently(self, button, timeout=0.5):
        """
        Check if a button was pressed within the specified timeout.
        
        Args:
            button (str): Button name
            timeout (float): Time window in seconds
        
        Returns:
            bool: True if the button was pressed within the timeout, False otherwise
        """
        if button not in self.last_press_time:
            return False
            
        return (time.time() - self.last_press_time[button]) < timeout
    
    def get_all_pressed(self):
        """
        Get list of all currently pressed buttons.
        
        Returns:
            list: List of button names that are currently pressed
        """
        pressed = []
        for button_name, button in self.buttons.items():
            try:
                if button.is_pressed:
                    pressed.append(button_name)
            except:
                pass
        return pressed
    
    def get_button_pins(self):
        """
        Get the current button pin configuration.
        
        Returns:
            dict: Dictionary mapping button names to GPIO pins
        """
        return self.button_pins.copy()
    
    def set_button_pin(self, button, pin):
        """
        Update a button's GPIO pin number.
        
        Args:
            button (str): Button name
            pin (int): GPIO pin number
        """
        # Clean up old button if it exists
        if button in self.buttons:
            old_button = self.buttons[button]
            old_button.close()
            
        # Create new button object
        self.button_pins[button] = pin
        new_button = Button(
            pin=pin, 
            pull_up=self.pull_up,
            bounce_time=self.bounce_time_s
        )
        
        # Set up callbacks
        new_button.when_pressed = lambda b=button: self._handle_button_pressed(b)
        new_button.when_released = lambda b=button: self._handle_button_released(b)
        
        # Store button
        self.buttons[button] = new_button
        print(f"Updated button {button} to GPIO pin {pin}")
    
    def cleanup(self):
        """Clean up GPIO resources"""
        self.stop_polling()
        print("Cleaning up GPIO resources...")
        for button in self.buttons.values():
            button.close()
        print("Cleanup complete")


# Example usage
if __name__ == "__main__":
    try:
        print("Button Connect Test")
        print("===================")
        print("Press buttons to see inputs (Ctrl+C to exit)")
        
        # Initialize with default pins - change this if needed for your hardware
        # pull_up=True means buttons should connect to GND when pressed
        # pull_up=False means buttons should connect to 3.3V when pressed
        buttons = ButtonConnect(pull_up=True)
        
        # Register a callback for all buttons
        def button_callback(button):
            print(f"CALLBACK: Button {button} was pressed!")
        
        buttons.register_callback('all', button_callback)
        
        # Start polling
        buttons.start_polling()
        
        print("\nWaiting for button presses...")
        print("Try pressing each button to test.")
        print("Currently monitoring buttons:", list(buttons.button_pins.keys()))
        print("Press Ctrl+C to exit\n")
        
        # Main loop - just monitor button states
        while True:
            pressed = buttons.get_all_pressed()
            if pressed:
                print(f"Currently pressed: {pressed}")
            time.sleep(0.2)  # Reduce console spam
    
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Clean up
        if 'buttons' in locals():
            buttons.cleanup() 