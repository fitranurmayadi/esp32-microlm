#include <stdio.h>
#include <string.h>
#include <string>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "esp_vfs_dev.h"
#include "driver/usb_serial_jtag.h"
#include "driver/usb_serial_jtag_vfs.h"
#include "kibo_inference.h"

extern "C" void app_main(void) {
    // Enable USB-Serial-JTAG for native USB boards (Nano ESP32, XIAO ESP32-S3)
    usb_serial_jtag_driver_config_t usb_serial_jtag_config = {
        .tx_buffer_size = 512,
        .rx_buffer_size = 512,
    };
    usb_serial_jtag_driver_install(&usb_serial_jtag_config);
    usb_serial_jtag_vfs_use_driver();
    usb_serial_jtag_vfs_set_rx_line_endings(ESP_LINE_ENDINGS_CRLF);
    usb_serial_jtag_vfs_set_tx_line_endings(ESP_LINE_ENDINGS_CRLF);
    
    // Configure standard I/O
    setvbuf(stdin, NULL, _IONBF, 0);
    setvbuf(stdout, NULL, _IONBF, 0);
    
    vTaskDelay(pdMS_TO_TICKS(1500));
    
    printf("\n");
    printf("==========================================================================\n");
    printf("  🤖 ESP32 MICRO-LM v4.0 (1.84M PARAMETER CAUSAL TRANSFORMER)             \n");
    printf("  • Core Engine: 4-Layer Causal Transformer (1.84M Parameters in PSRAM)   \n");
    printf("  • Precision:   W8A32 INT8 Quantization (Per-Tensor Symmetric Weights)   \n");
    printf("  • Compute:     Dual-Core Xtensa LX7 Parallel Execution (Core 0 + 1)     \n");
    printf("==========================================================================\n");
    
    if (!kibo_init_model()) {
        printf("❌ Fatal: Failed to initialize Kibo model in PSRAM!\n");
        return;
    }
    
    printf("\nType your message below and press Enter (Commands: 'benchmark', 'status'):\n\nUser: ");
    fflush(stdout);
    
    char line_buf[256];
    int line_idx = 0;
    while (true) {
        uint8_t ch = 0;
        int n = usb_serial_jtag_read_bytes(&ch, 1, 0);
        if (n <= 0) {
            int c = getchar();
            if (c != EOF) {
                ch = (uint8_t)c;
                n = 1;
            }
        }
        
        if (n > 0) {
            if (ch == '\r' || ch == '\n') {
                if (line_idx > 0) {
                    line_buf[line_idx] = '\0';
                    std::string input(line_buf);
                    kibo_process_chat(input);
                    line_idx = 0;
                }
            } else if (line_idx < (int)sizeof(line_buf) - 1) {
                line_buf[line_idx++] = (char)ch;
            }
        }
        vTaskDelay(pdMS_TO_TICKS(5));
    }
}
