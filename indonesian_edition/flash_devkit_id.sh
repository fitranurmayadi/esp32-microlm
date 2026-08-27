#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIRMWARE_DIR="$SCRIPT_DIR/kibo_esp32_id"

echo "[kibo-id] mengompilasi firmware edisi bahasa indonesia untuk devkit n16r8..."
arduino-cli compile --export-binaries \
  --fqbn "esp32:esp32:esp32s3:FlashSize=16M,FlashMode=dio,PSRAM=opi,PartitionScheme=app3M_fat9M_16MB,CDCOnBoot=default" \
  "$FIRMWARE_DIR"

BUILD_DIR="$FIRMWARE_DIR/build/esp32.esp32.esp32s3"
CORE_DIR="/home/aiot/.arduino15/packages/esp32/hardware/esp32/3.3.11"
MODEL_BIN="$FIRMWARE_DIR/kibo_model_int8.bin"

PORT=$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | head -n 1)
if [ -z "$PORT" ]; then
    echo "error: port serial tidak ditemukan" >&2
    exit 1
fi

echo "[kibo-id] target port: $PORT"
python3 -m esptool --chip esp32s3 --port "$PORT" --baud 921600 --before default-reset --after hard-reset write-flash \
  --flash-mode dio --flash-freq 80m --flash-size 16MB \
  0x0 "$BUILD_DIR/kibo_esp32_id.ino.bootloader.bin" \
  0x8000 "$BUILD_DIR/kibo_esp32_id.ino.partitions.bin" \
  0xe000 "$CORE_DIR/tools/partitions/boot_app0.bin" \
  0x10000 "$BUILD_DIR/kibo_esp32_id.ino.bin" \
  0x610000 "$MODEL_BIN"

echo "[kibo-id] flashing selesai"
