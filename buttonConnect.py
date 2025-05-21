from gpiozero import Button
from signal import pause
import time

# Default GPIO pin configuration - modify these values as needed for your hardware setup
# BCM pin numbering is used (not physical pin numbers)
GPIO_UP = 16       # Up button GPIO pin
GPIO_DOWN = 26     # Down button GPIO pin  
GPIO_LEFT = 20     # Left button GPIO pin
GPIO_RIGHT = 19    # Right button GPIO pin
GPIO_CAPTURE = 21   # Select/Enter button GPIO pin

# Button press callbacks and state tracking
button_pressed = {
    'up': False,
    'down': False,
    'left': False, 
    'right': False,
    'capture': False
}

last_press_time = {
    'up': 0,
    'down': 0,
    'left': 0,
    'right': 0,
    'capture': 0
}

# Global callback functions list
button_callbacks = {}

# Initialize the buttons
def init_buttons(pull_up=True, bounce_time=0.05):
    """
    Initialize all buttons with the specified configuration
    
    Args:
        pull_up (bool): True if buttons should use pull-up resistors (connect to GND when pressed)
                       False if buttons should use pull-down resistors (connect to 3.3V when pressed)
        bounce_time (float): Debounce time in seconds
    
    Returns:
        dict: Dictionary of button objects
    """
    print(f"Initializing buttons (pull_up={pull_up}, bounce_time={bounce_time})")
    
    buttons = {}
    button_pins = {
        'up': GPIO_UP,
        'down': GPIO_DOWN,
        'left': GPIO_LEFT,
        'right': GPIO_RIGHT,
        'capture': GPIO_CAPTURE
    }
    
    for name, pin in button_pins.items():
        try:
            print(f"Setting up {name} button on GPIO {pin}")
            # Create Button object with the specified parameters
            buttons[name] = Button(pin=pin, pull_up=pull_up, bounce_time=bounce_time)
            
            # Set up the button event handlers
            buttons[name].when_pressed = lambda n=name: handle_button_pressed(n)
            buttons[name].when_released = lambda n=name: handle_button_released(n)
            
            print(f"  {name} initialized successfully")
        except Exception as e:
            print(f"Error initializing {name} button on GPIO {pin}: {e}")
    
    return buttons

def handle_button_pressed(button_name):
    """
    Handler for button press events
    
    Args:
        button_name (str): Name of the button that was pressed
    """
    # Update button state
    button_pressed[button_name] = True
    last_press_time[button_name] = time.time()
    
    print(f"Button pressed: {button_name}")
    
    # Call registered callbacks
    if button_name in button_callbacks:
        button_callbacks[button_name](button_name)
    if 'all' in button_callbacks:
        button_callbacks['all'](button_name)

def handle_button_released(button_name):
    """
    Handler for button release events
    
    Args:
        button_name (str): Name of the button that was released
    """
    # Update button state
    button_pressed[button_name] = False
    print(f"Button released: {button_name}")

def register_callback(button_name, callback_function):
    """
    Register a callback function for a specific button
    
    Args:
        button_name (str): Name of button ('up', 'down', 'left', 'right', 'select', or 'all')
        callback_function (callable): Function to call when button is pressed
    """
    button_callbacks[button_name] = callback_function
    print(f"Registered callback for button: {button_name}")

def get_pressed_buttons():
    """
    Get list of all currently pressed buttons
    
    Returns:
        list: Names of buttons currently pressed
    """
    return [name for name, pressed in button_pressed.items() if pressed]

def was_pressed_recently(button_name, timeout=0.5):
    """
    Check if a button was pressed recently within the specified timeout
    
    Args:
        button_name (str): Name of the button to check
        timeout (float): Time window in seconds
        
    Returns:
        bool: True if button was pressed within the timeout period
    """
    if button_name not in last_press_time:
        return False
    
    return (time.time() - last_press_time[button_name]) < timeout

def cleanup():
    """Release all GPIO resources"""
    print("Cleaning up GPIO resources...")
    # The Button objects will be automatically cleaned up when they go out of scope
    # This is handled by gpiozero's Device.close() methods
    print("Cleanup complete")

# Example test function if this file is run directly
if __name__ == "__main__":
    try:
        print("Button Connect Test")
        print("===================")
        print("Press buttons to see inputs (Ctrl+C to exit)")
        
        # Initialize buttons - change pull_up based on your hardware setup
        # pull_up=True means buttons connect to GND when pressed
        # pull_up=False means buttons connect to 3.3V when pressed
        buttons = init_buttons(pull_up=True)
        
        # Register a test callback for all buttons
        def button_callback(button):
            print(f"CALLBACK: Button {button} was pressed!")
        
        register_callback('all', button_callback)
        
        print("\nWaiting for button presses...")
        print("Try pressing each button to test.")
        print("Press Ctrl+C to exit\n")
        
        # Keep the program running to receive button press events
        pause()  # This will wait until interrupted
        
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        cleanup() 