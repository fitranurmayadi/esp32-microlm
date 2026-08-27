#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | head -n 1)
if [ -z "$PORT" ]; then
    echo "error: no serial port (/dev/ttyUSB* or /dev/ttyACM*) detected" >&2
    exit 1
fi

python3 "$SCRIPT_DIR/kibo_esp32/serial_chat.py" "$PORT"
