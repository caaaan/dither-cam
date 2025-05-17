import RPi.GPIO as GPIO
import time
import threading

class ButtonConnect:
    """
    Interface for GPIO button connections on Raspberry Pi.
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
        # Configure GPIO
        GPIO.setmode(GPIO.BCM)  # Use BCM pin numbering
        
        # Default button pins if none provided
        self.button_pins = button_pins or {
            'up': 17,
            'down': 18,
            'left': 22,
            'right': 23,
            'select': 27
        }
        
        # Button state tracking
        self.button_states = {button: self.BUTTON_RELEASED for button in self.button_pins}
        self.last_press_time = {button: 0 for button in self.button_pins}
        self.callbacks = {}
        
        # Setup parameters
        self.bounce_time = bounce_time
        self.pull_up_down = GPIO.PUD_UP if pull_up else GPIO.PUD_DOWN
        self.input_event = GPIO.FALLING if pull_up else GPIO.RISING
        
        # Set up GPIO pins for buttons
        for button, pin in self.button_pins.items():
            GPIO.setup(pin, GPIO.IN, pull_up_down=self.pull_up_down)
            GPIO.add_event_detect(pin, self.input_event, callback=lambda channel, btn=button: self._handle_button_event(btn), 
                                  bouncetime=self.bounce_time)
                                  
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
            for button, pin in self.button_pins.items():
                # Read current state (pull-up means button pressed when pin reads LOW)
                current_state = GPIO.LOW if GPIO.input(pin) == GPIO.LOW else GPIO.HIGH
                button_pressed = (current_state == GPIO.LOW) if self.pull_up_down == GPIO.PUD_UP else (current_state == GPIO.HIGH)
                
                # Update button state
                self.button_states[button] = self.BUTTON_PRESSED if button_pressed else self.BUTTON_RELEASED
                
            time.sleep(interval)
    
    def _handle_button_event(self, button):
        """
        Handle button press/release events and invoke callbacks.
        
        Args:
            button (str): Button name
        """
        # Update button press time
        current_time = time.time()
        self.last_press_time[button] = current_time
        
        # Update state
        self.button_states[button] = self.BUTTON_PRESSED
        
        # Call button-specific callback if registered
        if button in self.callbacks:
            self.callbacks[button](button)
        
        # Call general callback if registered
        if 'all' in self.callbacks:
            self.callbacks['all'](button)
    
    def register_callback(self, button, callback_function):
        """
        Register a callback function for a specific button.
        
        Args:
            button (str): Button name or 'all' for all buttons
            callback_function (callable): Function to call when button is pressed
        """
        self.callbacks[button] = callback_function
    
    def is_pressed(self, button):
        """
        Check if a button is currently pressed.
        
        Args:
            button (str): Button name
        
        Returns:
            bool: True if the button is pressed, False otherwise
        """
        if button not in self.button_pins:
            return False
        
        pin = self.button_pins[button]
        state = GPIO.input(pin)
        
        # For pull-up, pressed is LOW; for pull-down, pressed is HIGH
        return state == GPIO.LOW if self.pull_up_down == GPIO.PUD_UP else state == GPIO.HIGH
    
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
        return [button for button, pin in self.button_pins.items() if self.is_pressed(button)]
    
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
        # Clean up old pin if it exists
        if button in self.button_pins:
            old_pin = self.button_pins[button]
            GPIO.remove_event_detect(old_pin)
            
        # Set up new pin
        self.button_pins[button] = pin
        GPIO.setup(pin, GPIO.IN, pull_up_down=self.pull_up_down)
        GPIO.add_event_detect(pin, self.input_event, callback=lambda channel, btn=button: self._handle_button_event(btn), 
                              bouncetime=self.bounce_time)
    
    def cleanup(self):
        """Clean up GPIO resources"""
        self.stop_polling()
        GPIO.cleanup([pin for pin in self.button_pins.values()])


# Example usage
if __name__ == "__main__":
    try:
        print("Button Connect Test")
        print("===================")
        print("Press buttons to see inputs (Ctrl+C to exit)")
        
        # Initialize with default pins
        buttons = ButtonConnect()
        
        # Register a callback for all buttons
        def button_callback(button):
            print(f"PRESSED: {button}")
        
        buttons.register_callback('all', button_callback)
        
        # Start polling
        buttons.start_polling()
        
        # Main loop - just monitor button states
        while True:
            pressed = buttons.get_all_pressed()
            if pressed:
                print(f"Currently pressed: {pressed}")
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Clean up
        if 'buttons' in locals():
            buttons.cleanup() 