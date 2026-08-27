#include <Arduino.h>
#include "kibo_inference.h"

// Forward declarations
void setup(void);
void loop(void);

void setup(void) {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\nkibo-mcu [1.84M params | int8 | xtensa-lx7 @ 240MHz]");
    Serial.printf("heap: %d bytes | psram: %d bytes\n", ESP.getFreeHeap(), ESP.getFreePsram());
    
    if (!kibo_init_model()) {
        Serial.println("error: failed to initialize model");
        return;
    }
    
    Serial.println("ready. enter prompt (e.g. 'hello kibo', 'what is 25 times 4')\n");
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
