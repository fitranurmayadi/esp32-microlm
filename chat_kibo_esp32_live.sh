#!/bin/bash
PORT=$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | head -n 1)
if [ -z "$PORT" ]; then
    echo "error: no serial port (/dev/ttyUSB* or /dev/ttyACM*) detected" >&2
    exit 1
fi

python3 /home/aiot/Projects/SBC/kibo_esp32/serial_chat.py "$PORT"
