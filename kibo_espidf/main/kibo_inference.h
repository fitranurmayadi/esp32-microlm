#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string>
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "kibo_vocab.h"

#define MODEL_MAGIC 0x4B49424F // 'KIBO'
#define N_LAYER 4
#define N_EMBD 192
#define N_HEAD 4
#define HEAD_DIM (N_EMBD / N_HEAD) // 48
#define PLE_DIM 128
#define VOCAB_SIZE KIBO_VOCAB_SIZE
#define MAX_SEQ_LEN 128

// Google Gemma 3n Per-Layer Embedding Adapter
struct KiboPLEAdapter {
    const int8_t* ple_gate_w;
    float ple_gate_scale;
    const int8_t* ple_proj_w;
    float ple_proj_scale;
    float ple_norm_w[N_EMBD];
    bool has_ple;
};

struct TransformerBlock {
    KiboPLEAdapter ple;
    
    float ln1_w[N_EMBD];
    float ln1_b[N_EMBD];
    
    const int8_t* qkv_w;
    float qkv_scale;
    float qkv_b[3 * N_EMBD];
    
    const int8_t* proj_w;
    float proj_scale;
    float proj_b[N_EMBD];
    
    float ln2_w[N_EMBD];
    float ln2_b[N_EMBD];
    
    const int8_t* fc_w;
    float fc_scale;
    float fc_b[4 * N_EMBD];
    
    const int8_t* mlp_proj_w;
    float mlp_proj_scale;
    float mlp_proj_b[N_EMBD];
};

struct KiboModel {
    const int8_t* tok_emb_w;
    float tok_emb_scale;
    const int8_t* pos_emb_w;
    float pos_emb_scale;
    
    // Gemma 3n Flash-Resident PLE Table Pointer [VOCAB_SIZE, N_LAYER * PLE_DIM]
    const int8_t* ple_table_w;
    float ple_table_scale;
    
    TransformerBlock blocks[N_LAYER];
    
    float ln_f_w[N_EMBD];
    float ln_f_b[N_EMBD];
    const int8_t* head_w;
    float head_scale;
};

struct KiboTelemetry {
    uint32_t total_tokens_generated;
    float last_generation_time_s;
    float last_tokens_per_sec;
    uint32_t free_sram_bytes;
    uint32_t free_psram_bytes;
    uint32_t cpu_freq_mhz;
    bool dual_core_active;
    bool ple_hybrid_active;
};

extern KiboModel kibo_model;
extern KiboTelemetry kibo_telemetry;

bool kibo_init_model();
void kibo_init_dual_core();
void kibo_process_chat(const std::string& user_input);
void run_live_mcu_benchmark();
