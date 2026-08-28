#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string>
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define MODEL_MAGIC 0x4D4C4D38
#define N_LAYER 4
#define N_EMBD 192
#define N_HEAD 4
#define HEAD_DIM (N_EMBD / N_HEAD) // 48
#define VOCAB_SIZE 91
#define MAX_SEQ_LEN 128

struct TransformerBlock {
    const float* ln1_w;
    const float* ln1_b;
    const int8_t* qkv_w;
    float qkv_scale;
    const float* qkv_b;
    
    const float* ln2_w;
    const float* ln2_b;
    const int8_t* fc_w;
    float fc_scale;
    const float* fc_b;
    
    const int8_t* proj_w;
    float proj_scale;
    const float* proj_b;
};

struct KiboModel {
    const int8_t* tok_emb_w;
    float tok_emb_scale;
    const int8_t* pos_emb_w;
    float pos_emb_scale;
    
    TransformerBlock blocks[N_LAYER];
    
    const float* ln_f_w;
    const float* ln_f_b;
    const int8_t* head_w;
    float head_scale;
};

struct KiboTelemetry {
    uint32_t total_tokens;
    float avg_tokens_per_sec;
    float last_response_time_sec;
    uint32_t free_sram_bytes;
    uint32_t free_psram_bytes;
    uint32_t cpu_freq_mhz;
    bool dual_core_active;
    bool pie_simd_active;
};

extern KiboModel kibo_model;
extern KiboTelemetry kibo_telemetry;

bool kibo_init_model();
void kibo_init_dual_core();
void kibo_process_chat(const std::string& user_input);
void run_live_mcu_benchmark();
