#include <Arduino.h>
#include "kibo_inference.h"

// Forward declarations
void setup(void);
void loop(void);

void setup(void) {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\nesp32 micro-lm v2.0 [1.84M params | int8 w8a32 | dual-core 240MHz | edisi-indonesia]");
    Serial.printf("sram: %d bytes | psram: %d bytes\n", ESP.getFreeHeap(), ESP.getFreePsram());
    
    if (!kibo_init_model()) {
        Serial.println("error: gagal menginisialisasi model");
        return;
    }
    
    Serial.println("siap. masukkan prompt (contoh: 'halo', 'hitung 25 kali 4', 'status', 'benchmark')\n");
    Serial.print("User: ");
}

static String input_buffer = "";

void loop(void) {
    while (Serial.available() > 0) {
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') {
            input_buffer.trim();
            if (input_buffer.length() > 0) {
                Serial.printf("\nUser: %s\n", input_buffer.c_str());
                kibo_process_chat(input_buffer);
                input_buffer = "";
            }
        } else {
            input_buffer += c;
        }
    }
    delay(5);
}
