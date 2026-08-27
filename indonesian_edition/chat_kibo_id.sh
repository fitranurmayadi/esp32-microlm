#!/bin/bash
PORT=$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | head -n 1)
if [ -z "$PORT" ]; then
    echo "error: port serial (/dev/ttyUSB* atau /dev/ttyACM*) tidak ditemukan" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/kibo_esp32_id/serial_chat_id.py" "$PORT"
