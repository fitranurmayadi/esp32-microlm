#pragma once
#include <Arduino.h>
#include "esp_partition.h"
#include "esp_heap_caps.h"
#include <math.h>
#include "kibo_vocab.h"

#define MODEL_MAGIC 0x4B49424F // 'KIBO'

#define N_EMBD 192
#define N_HEAD 4
#define HEAD_DIM (N_EMBD / N_HEAD) // 48
#define N_LAYER 4
#define MAX_SEQ_LEN 128
#define VOCAB_SIZE KIBO_VOCAB_SIZE

struct TransformerBlock {
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
    
    TransformerBlock blocks[N_LAYER];
    
    float ln_f_w[N_EMBD];
    float ln_f_b[N_EMBD];
    
    const int8_t* head_w;
    float head_scale;
};

extern KiboModel kibo_model;

bool kibo_init_model();
void kibo_process_chat(const String& user_input);
