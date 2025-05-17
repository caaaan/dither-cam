#!/bin/bash

# Launch script for dither-cam application
# DONT FORGET TO RUN "chmod +x launch.sh" TO MAKE THE FILE EXECUTABLE
show_help() {
    echo "Usage: ./launch.sh [OPTION]"
    echo "Launch the dither-cam application in different configurations."
    echo ""
    echo "Options:"
    echo "  -w, --window      Launch in 480x320 window mode (default)"
    echo "  -f, --fullscreen  Launch in fullscreen mode"
    echo "  -r, --resolution  Specify resolution (format: WIDTHxHEIGHT, e.g. 800x600)"
    echo "  -h, --help        Display this help and exit"
    echo ""
    echo "Examples:"
    echo "  ./launch.sh                   # Launch in default window mode (480x320)"
    echo "  ./launch.sh -f                # Launch in fullscreen mode"
    echo "  ./launch.sh -r 800x600        # Launch in custom resolution"
    echo "  ./launch.sh -f -r 1024x768    # Launch in fullscreen with specified resolution"
}

# Default values
FULLSCREEN=false
RESOLUTION="480x320"


while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f|--fullscreen) FULLSCREEN=true ;;
        -w|--window) FULLSCREEN=false ;;
        -r|--resolution) RESOLUTION="$2"; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "Unknown parameter: $1"; show_help; exit 1 ;;
    esac
    shift
done

CMD="source venv/bin/activate && python3 main.py --resolution $RESOLUTION"
if [ "$FULLSCREEN" = true ]; then
    CMD="$CMD --fullscreen"
fi

echo "Launching dither-cam with command: $CMD"
$CMD 