#!/bin/bash
set -e

echo "[kibo] flashing firmware to arduino nano esp32 (16mb flash / 8mb psram)"

BUILD_DIR="/home/aiot/Projects/SBC/kibo_esp32/build/arduino.esp32.nano_nora"
CORE_DIR="/home/aiot/.arduino15/packages/esp32/hardware/esp32/3.3.11"
MODEL_BIN="/home/aiot/Projects/SBC/kibo_microlm/kibo_model_int8.bin"

PORT=$(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | head -n 1)
if [ -z "$PORT" ]; then
    echo "error: no serial port detected" >&2
    exit 1
fi

echo "[kibo] target port: $PORT"
python3 -m esptool --chip esp32s3 --port "$PORT" --baud 921600 write-flash \
  --flash-mode dio --flash-freq 80m --flash-size 16MB \
  0x0 "$BUILD_DIR/kibo_esp32.ino.bootloader.bin" \
  0x8000 "$BUILD_DIR/kibo_esp32.ino.partitions.bin" \
  0xe000 "$CORE_DIR/tools/partitions/boot_app0.bin" \
  0x10000 "$BUILD_DIR/kibo_esp32.ino.bin" \
  0x610000 "$MODEL_BIN"

echo "[kibo] flashing complete"
