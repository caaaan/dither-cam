#!/bin/bash

# Launch script for dither-cam application
# DONT FORGET TO RUN "chmod +x launch.sh" TO MAKE THE FILE EXECUTABLE
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/logs/dithercam.log"

show_help() {
    echo "Usage: ./launch.sh [OPTION]"
    echo "Launch the dither-cam application in different configurations."
    echo ""
    echo "Options:"
    echo "  -w, --window           Launch in 480x320 window mode (default)"
    echo "  -f, --fullscreen       Launch in fullscreen mode"
    echo "  -r, --resolution WxH   Window resolution (e.g. 800x600)"
    echo "  -s, --source SOURCE    Camera source: auto, picamera, webcam (default: auto)"
    echo "  -d, --device DEV       Webcam device index or path (default: 0)"
    echo "  -l, --logs             Tail the log file instead of launching"
    echo "  -h, --help             Display this help and exit"
    echo ""
    echo "Log file: $LOG_FILE"
    echo ""
    echo "Examples:"
    echo "  ./launch.sh                        # Auto-detect camera, 480x320 window"
    echo "  ./launch.sh -f                     # Fullscreen"
    echo "  ./launch.sh --source webcam        # Force USB webcam"
    echo "  ./launch.sh --source webcam -d 1   # USB webcam on /dev/video1"
    echo "  ./launch.sh --source picamera      # Force CSI ribbon camera"
    echo "  ./launch.sh --logs                 # Watch live log output"
}

# Default values
FULLSCREEN=false
RESOLUTION="480x320"
SOURCE="auto"
DEVICE="0"
export DISPLAY=:0

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f|--fullscreen) FULLSCREEN=true ;;
        -w|--window) FULLSCREEN=false ;;
        -r|--resolution) RESOLUTION="$2"; shift ;;
        -s|--source) SOURCE="$2"; shift ;;
        -d|--device) DEVICE="$2"; shift ;;
        -l|--logs)
            if [ -f "$LOG_FILE" ]; then
                tail -f "$LOG_FILE"
            else
                echo "No log file yet: $LOG_FILE"
                echo "Run the app first."
            fi
            exit 0
            ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "Unknown parameter: $1"; show_help; exit 1 ;;
    esac
    shift
done

# Build Python arguments
ARGS="--resolution $RESOLUTION --source $SOURCE --device $DEVICE"
if [ "$FULLSCREEN" = true ]; then
    ARGS="$ARGS --fullscreen"
fi

# Display what we're going to run
echo "Launching dither-cam  resolution=$RESOLUTION  fullscreen=$FULLSCREEN  source=$SOURCE"
echo "Log file: $LOG_FILE"

# Activate venv if present
if [ -d "$SCRIPT_DIR/venv" ] && [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$SCRIPT_DIR/venv/bin/activate"
fi

python3 "$SCRIPT_DIR/main.py" $ARGS
