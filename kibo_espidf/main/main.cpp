#include <stdio.h>
#include <string.h>
#include <string>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "esp_vfs_dev.h"
#include "kibo_inference.h"

extern "C" void app_main(void) {
    // Configure standard I/O for UART
    setvbuf(stdin, NULL, _IONBF, 0);
    setvbuf(stdout, NULL, _IONBF, 0);
    
    vTaskDelay(pdMS_TO_TICKS(1500));
    
    printf("\n");
    printf("==========================================================================\n");
    printf("  🤖 KIBO MICRO-LM v3.0 (ESP-IDF NATIVE PIE SIMD EDITION)                 \n");
    printf("  • Core Engine: 4-Layer Causal Transformer (1.84M Parameters, INT8 W8A32)\n");
    printf("  • Hardware:    ESP32-S3 Dual-Core Xtensa LX7 @ 240MHz (Core 0 + Core 1) \n");
    printf("  • Coprocessor: CP0 128-bit Vector SIMD Enabled (CONFIG_ESP32S3_COPROC)  \n");
    printf("==========================================================================\n");
    
    if (!kibo_init_model()) {
        printf("❌ Fatal: Failed to initialize Kibo model in PSRAM!\n");
        return;
    }
    
    printf("\nType your message below and press Enter (Commands: 'benchmark', 'status'):\n\nUser: ");
    fflush(stdout);
    
    char line_buf[256];
    while (true) {
        if (fgets(line_buf, sizeof(line_buf), stdin) != NULL) {
            std::string input(line_buf);
            if (!input.empty() && input != "\n" && input != "\r\n") {
                kibo_process_chat(input);
            }
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}
