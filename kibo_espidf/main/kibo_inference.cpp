#include "kibo_inference.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_system.h"
#include "esp_chip_info.h"

extern "C" {
    extern const uint8_t kibo_embedded_model_start[];
    extern const uint8_t kibo_embedded_model_end[];
}

KiboModel kibo_model;
KiboTelemetry kibo_telemetry = {0, 0.0f, 0.0f, 0, 0, 240, false, true};

// SRAM Hot Working Buffers
static float* act_x = NULL;
static float* act_xb = NULL;
static float* act_qkv = NULL;
static float* act_att = NULL;
static float* act_mlp = NULL;
static float* act_proj = NULL;
static float* act_mlp_proj = NULL;
static float* act_logits = NULL;

// Octal PSRAM Dynamic KV-Cache
static float* kv_k_cache = NULL;
static float* kv_v_cache = NULL;

static const uint8_t* mmap_base = NULL;

// ============================================================================
// FreeRTOS Dual-Core Parallel Execution Engine
// ============================================================================
enum Core0JobType {
    JOB_IDLE = 0,
    JOB_MATMUL_SLICE
};

struct Core0Job {
    volatile Core0JobType type;
    float* out;
    const float* x;
    const int8_t* w;
    float scale;
    const float* bias;
    int start_row;
    int end_row;
    int cols;
};

static Core0Job core0_job;
static TaskHandle_t core0_task_handle = NULL;
static TaskHandle_t main_task_handle = NULL;

static void safe_flash_copy(void* dest, const void* src, size_t n) {
    uint8_t* d = (uint8_t*)dest;
    const uint8_t* s = (const uint8_t*)src;
    for (size_t i = 0; i < n; i++) {
        d[i] = s[i];
    }
}

static inline uint32_t read_u32(const uint8_t* p) {
    uint32_t v = 0;
    safe_flash_copy(&v, p, sizeof(v));
    return v;
}

static inline int32_t read_i32(const uint8_t* p) {
    int32_t v = 0;
    safe_flash_copy(&v, p, sizeof(v));
    return v;
}

static inline float read_f32(const uint8_t* p) {
    float v = 0.0f;
    safe_flash_copy(&v, p, sizeof(v));
    return v;
}

// 4-Way Vectorized Layer Normalization
static inline void layer_norm(float* out, const float* x, const float* w, const float* b, int dim) {
    float mean0 = 0.0f, mean1 = 0.0f, mean2 = 0.0f, mean3 = 0.0f;
    for (int i = 0; i < dim; i += 4) {
        mean0 += x[i+0]; mean1 += x[i+1]; mean2 += x[i+2]; mean3 += x[i+3];
    }
    float mean = (mean0 + mean1 + mean2 + mean3) / dim;
    
    float var0 = 0.0f, var1 = 0.0f, var2 = 0.0f, var3 = 0.0f;
    for (int i = 0; i < dim; i += 4) {
        float d0 = x[i+0] - mean; float d1 = x[i+1] - mean;
        float d2 = x[i+2] - mean; float d3 = x[i+3] - mean;
        var0 += d0*d0; var1 += d1*d1; var2 += d2*d2; var3 += d3*d3;
    }
    float var = (var0 + var1 + var2 + var3) / dim;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    
    for (int i = 0; i < dim; i += 4) {
        out[i+0] = (x[i+0] - mean) * inv_std * w[i+0] + b[i+0];
        out[i+1] = (x[i+1] - mean) * inv_std * w[i+1] + b[i+1];
        out[i+2] = (x[i+2] - mean) * inv_std * w[i+2] + b[i+2];
        out[i+3] = (x[i+3] - mean) * inv_std * w[i+3] + b[i+3];
    }
}

// Fast Sigmoid GeLU: x * sigmoid(1.702 * x)
static inline float gelu_act(float x) {
    return x / (1.0f + expf(-1.702f * x));
}

// 8-Way Direct FPU Pipeline Unrolling (Zero-Overhead Pointer Indexing)
static inline void matmul_int8_slice(float* out, const float* x, const int8_t* w, float scale, const float* bias, int start_row, int end_row, int cols) {
    for (int r = start_row; r < end_row; r++) {
        float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
        float dot4 = 0.0f, dot5 = 0.0f, dot6 = 0.0f, dot7 = 0.0f;
        const int8_t* w_row = w + r * cols;
        int c = 0;
        
        for (; c <= cols - 8; c += 8) {
            dot0 += x[c + 0] * (float)w_row[c + 0];
            dot1 += x[c + 1] * (float)w_row[c + 1];
            dot2 += x[c + 2] * (float)w_row[c + 2];
            dot3 += x[c + 3] * (float)w_row[c + 3];
            dot4 += x[c + 4] * (float)w_row[c + 4];
            dot5 += x[c + 5] * (float)w_row[c + 5];
            dot6 += x[c + 6] * (float)w_row[c + 6];
            dot7 += x[c + 7] * (float)w_row[c + 7];
        }
        
        float dot = (dot0 + dot1) + (dot2 + dot3) + (dot4 + dot5) + (dot6 + dot7);
        for (; c < cols; c++) {
            dot += x[c] * (float)w_row[c];
        }
        
        float sum = dot * scale;
        if (bias != NULL) {
            sum += bias[r];
        }
        out[r] = sum;
    }
}

// Core 1 Dedicated FreeRTOS Worker Task
static void kibo_core1_worker_task(void* param) {
    while (true) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        if (core0_job.type == JOB_MATMUL_SLICE) {
            matmul_int8_slice(
                core0_job.out,
                core0_job.x,
                core0_job.w,
                core0_job.scale,
                core0_job.bias,
                core0_job.start_row,
                core0_job.end_row,
                core0_job.cols
            );
            core0_job.type = JOB_IDLE;
        }
        if (main_task_handle != NULL) {
            xTaskNotifyGive(main_task_handle);
        }
    }
}

void kibo_init_dual_core() {
    if (core0_task_handle == NULL) {
        main_task_handle = xTaskGetCurrentTaskHandle();
        BaseType_t ret = xTaskCreatePinnedToCore(
            kibo_core1_worker_task,
            "kibo_core1_worker",
            4096,
            NULL,
            5,
            &core0_task_handle,
            1 // Pin to Core 1 (APP_CPU) while app_main runs on Core 0 (PRO_CPU)
        );
        if (ret == pdPASS) {
            kibo_telemetry.dual_core_active = true;
            printf("[kibo-idf] dual-core parallel engine active (Core 0 + Core 1 @ 240MHz)\n");
        } else {
            kibo_telemetry.dual_core_active = false;
            printf("[kibo-idf] warning: running in single-core mode\n");
        }
    }
}

// Parallel / Single-Core Dispatcher
static void matmul_int8_vec(float* out, const float* x, const int8_t* w, float scale, const float* bias, int rows, int cols) {
    if (kibo_telemetry.dual_core_active && core0_task_handle != NULL && rows >= 512) {
        int mid = rows / 2;
        main_task_handle = xTaskGetCurrentTaskHandle();
        
        // Dispatch first half to Core 0
        core0_job.type = JOB_MATMUL_SLICE;
        core0_job.out = out;
        core0_job.x = x;
        core0_job.w = w;
        core0_job.scale = scale;
        core0_job.bias = bias;
        core0_job.start_row = 0;
        core0_job.end_row = mid;
        core0_job.cols = cols;
        xTaskNotifyGive(core0_task_handle);
        
        // Compute second half on Core 1 concurrently
        matmul_int8_slice(out, x, w, scale, bias, mid, rows, cols);
        
        // Wait for Core 0 completion
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    } else {
        matmul_int8_slice(out, x, w, scale, bias, 0, rows, cols);
    }
}

static int effective_max_seq_len = MAX_SEQ_LEN;

static void forward_token(int token, int pos, int seq_len) {
    for (int i = 0; i < N_EMBD; i++) {
        float tok_val = ((float)kibo_model.tok_emb_w[token * N_EMBD + i]) * kibo_model.tok_emb_scale;
        float pos_val = ((float)kibo_model.pos_emb_w[pos * N_EMBD + i]) * kibo_model.pos_emb_scale;
        act_x[i] = tok_val + pos_val;
    }
    
    for (int l = 0; l < N_LAYER; l++) {
        TransformerBlock& blk = kibo_model.blocks[l];
        
        layer_norm(act_xb, act_x, blk.ln1_w, blk.ln1_b, N_EMBD);
        matmul_int8_vec(act_qkv, act_xb, blk.qkv_w, blk.qkv_scale, blk.qkv_b, 3 * N_EMBD, N_EMBD);
        
        float* q = act_qkv;
        float* k = act_qkv + N_EMBD;
        float* v = act_qkv + 2 * N_EMBD;
        
        int kv_offset = (l * effective_max_seq_len + pos) * N_EMBD;
        for (int i = 0; i < N_EMBD; i++) {
            kv_k_cache[kv_offset + i] = k[i];
            kv_v_cache[kv_offset + i] = v[i];
        }
        
        float* att_out = act_xb;
        for (int i = 0; i < N_EMBD; i++) att_out[i] = 0.0f;
        
        float scale = 1.0f / sqrtf((float)HEAD_DIM);
        
        for (int h = 0; h < N_HEAD; h++) {
            const float* q_h = q + h * HEAD_DIM;
            float* att_scores = act_att + h * effective_max_seq_len;
            
            float max_score = -1e9f;
            for (int t = 0; t <= pos; t++) {
                int prev_kv_offset = (l * effective_max_seq_len + t) * N_EMBD + h * HEAD_DIM;
                const float* k_h = kv_k_cache + prev_kv_offset;
                
                float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
                for (int d = 0; d < HEAD_DIM; d += 4) {
                    dot0 += q_h[d+0] * k_h[d+0];
                    dot1 += q_h[d+1] * k_h[d+1];
                    dot2 += q_h[d+2] * k_h[d+2];
                    dot3 += q_h[d+3] * k_h[d+3];
                }
                float dot = (dot0 + dot1 + dot2 + dot3) * scale;
                att_scores[t] = dot;
                if (dot > max_score) max_score = dot;
            }
            
            float sum_exp = 0.0f;
            for (int t = 0; t <= pos; t++) {
                att_scores[t] = expf(att_scores[t] - max_score);
                sum_exp += att_scores[t];
            }
            float inv_sum = 1.0f / sum_exp;
            for (int t = 0; t <= pos; t++) {
                att_scores[t] *= inv_sum;
            }
            
            float* head_out = att_out + h * HEAD_DIM;
            for (int t = 0; t <= pos; t++) {
                int prev_v_offset = (l * effective_max_seq_len + t) * N_EMBD + h * HEAD_DIM;
                const float* v_h = kv_v_cache + prev_v_offset;
                float a_t = att_scores[t];
                for (int d = 0; d < HEAD_DIM; d += 4) {
                    head_out[d+0] += a_t * v_h[d+0];
                    head_out[d+1] += a_t * v_h[d+1];
                    head_out[d+2] += a_t * v_h[d+2];
                    head_out[d+3] += a_t * v_h[d+3];
                }
            }
        }
        
        matmul_int8_vec(act_proj, att_out, blk.proj_w, blk.proj_scale, blk.proj_b, N_EMBD, N_EMBD);
        for (int i = 0; i < N_EMBD; i++) {
            act_x[i] += act_proj[i];
        }
        
        layer_norm(act_xb, act_x, blk.ln2_w, blk.ln2_b, N_EMBD);
        matmul_int8_vec(act_mlp, act_xb, blk.fc_w, blk.fc_scale, blk.fc_b, 4 * N_EMBD, N_EMBD);
        for (int i = 0; i < 4 * N_EMBD; i++) {
            act_mlp[i] = gelu_act(act_mlp[i]);
        }
        
        matmul_int8_vec(act_mlp_proj, act_mlp, blk.mlp_proj_w, blk.mlp_proj_scale, blk.mlp_proj_b, N_EMBD, 4 * N_EMBD);
        for (int i = 0; i < N_EMBD; i++) {
            act_x[i] += act_mlp_proj[i];
        }
    }
    
    layer_norm(act_xb, act_x, kibo_model.ln_f_w, kibo_model.ln_f_b, N_EMBD);
    matmul_int8_slice(act_logits, act_xb, kibo_model.head_w, kibo_model.head_scale, NULL, 0, VOCAB_SIZE, N_EMBD);
}

static int sample_token(float* logits, int vocab_size, float temperature) {
    int best_idx = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_idx = i;
        }
    }
    return best_idx;
}

static int tokenize_text(const char* text, int* tokens, int max_tokens) {
    int count = 0;
    int len = strlen(text);
    int i = 0;
    
    while (i < len && count < max_tokens) {
        int best_token = -1;
        int best_len = 0;
        
        for (int v = 0; v < KIBO_VOCAB_SIZE; v++) {
            const char* entry = KIBO_VOCAB_TABLE[v];
            int elen = strlen(entry);
            if (elen > 0 && strncmp(&text[i], entry, elen) == 0) {
                if (elen > best_len) {
                    best_len = elen;
                    best_token = v;
                }
            }
        }
        
        if (best_token != -1) {
            tokens[count++] = best_token;
            i += best_len;
        } else {
            tokens[count++] = 1; // <UNK>
            i++;
        }
    }
    return count;
}

static bool kibo_math_calculator(const std::string& input, double& out_result, std::string& out_op_str) {
    std::string text = input;
    for (char& c : text) c = tolower(c);
    
    double num1 = 0, num2 = 0;
    char op = 0;
    bool found_op = false;
    
    // Check multiplication
    size_t pos = text.find("kali");
    if (pos == std::string::npos) pos = text.find("times");
    if (pos == std::string::npos) pos = text.find("*");
    if (pos == std::string::npos) pos = text.find("x");
    if (pos != std::string::npos && ((pos > 0 && isdigit((unsigned char)text[pos-1])) || (text.find(" ") != std::string::npos))) {
        op = '*'; found_op = true;
    }
    
    // Check addition
    if (!found_op) {
        pos = text.find("tambah");
        if (pos == std::string::npos) pos = text.find("plus");
        if (pos == std::string::npos) pos = text.find("+");
        if (pos != std::string::npos) { op = '+'; found_op = true; }
    }
    
    // Check subtraction
    if (!found_op) {
        pos = text.find("kurang");
        if (pos == std::string::npos) pos = text.find("minus");
        if (pos == std::string::npos) pos = text.find("-");
        if (pos != std::string::npos) { op = '-'; found_op = true; }
    }
    
    // Check division
    if (!found_op) {
        pos = text.find("bagi");
        if (pos == std::string::npos) pos = text.find("divided");
        if (pos == std::string::npos) pos = text.find("/");
        if (pos != std::string::npos) { op = '/'; found_op = true; }
    }
    
    if (!found_op) return false;
    
    char str1[64] = {0}; char str2[64] = {0};
    int s1_idx = 0, s2_idx = 0;
    
    for (int i = (int)pos - 1; i >= 0; i--) {
        if (isdigit((unsigned char)text[i]) || text[i] == '.') {
            for (int j = i; j >= 0; j--) {
                if (isdigit((unsigned char)text[j]) || text[j] == '.') {
                    str1[s1_idx++] = text[j];
                } else if (s1_idx > 0) break;
            }
            break;
        }
    }
    
    for (size_t i = pos + 1; i < text.length(); i++) {
        if (isdigit((unsigned char)text[i]) || text[i] == '.') {
            for (size_t j = i; j < text.length(); j++) {
                if (isdigit((unsigned char)text[j]) || text[j] == '.') {
                    str2[s2_idx++] = text[j];
                } else if (s2_idx > 0) break;
            }
            break;
        }
    }
    
    if (s1_idx == 0 || s2_idx == 0) return false;
    
    for (int i = 0; i < s1_idx / 2; i++) {
        char temp = str1[i]; str1[i] = str1[s1_idx - 1 - i]; str1[s1_idx - 1 - i] = temp;
    }
    
    num1 = atof(str1);
    num2 = atof(str2);
    
    if (op == '+') out_result = num1 + num2;
    else if (op == '-') out_result = num1 - num2;
    else if (op == '*') out_result = num1 * num2;
    else if (op == '/') out_result = (num2 != 0) ? (num1 / num2) : 0;
    
    char buf[128];
    snprintf(buf, sizeof(buf), "%.2f %c %.2f = %.2f", num1, op, num2, out_result);
    out_op_str = buf;
    return true;
}

bool kibo_init_model() {
    printf("\n[kibo-idf] initializing v3.0 native esp-idf engine...\n");
    
    // Allocate high-speed internal SRAM buffers
    act_x        = (float*)heap_caps_malloc(N_EMBD * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    act_xb       = (float*)heap_caps_malloc(N_EMBD * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    act_qkv      = (float*)heap_caps_malloc(3 * N_EMBD * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    act_att      = (float*)heap_caps_malloc(N_HEAD * MAX_SEQ_LEN * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    act_mlp      = (float*)heap_caps_malloc(4 * N_EMBD * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    act_proj     = (float*)heap_caps_malloc(N_EMBD * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    act_mlp_proj = (float*)heap_caps_malloc(N_EMBD * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    act_logits   = (float*)heap_caps_malloc(VOCAB_SIZE * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    
    effective_max_seq_len = MAX_SEQ_LEN;
    size_t kv_size = N_LAYER * effective_max_seq_len * N_EMBD * sizeof(float);
    kv_k_cache = (float*)heap_caps_malloc(kv_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    kv_v_cache = (float*)heap_caps_malloc(kv_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    
    if (!act_x || !act_qkv || !kv_k_cache || !kv_v_cache) {
        printf("[kibo-idf] error: failed to allocate working buffers\n");
        return false;
    }
    
    kibo_init_dual_core();

    const esp_partition_t* part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_DATA_FAT, NULL);
    size_t model_size = (+kibo_embedded_model_end > +kibo_embedded_model_start) ? (size_t)(kibo_embedded_model_end - kibo_embedded_model_start) : 2 * 1024 * 1024;
    
    if (part != NULL && part->size < model_size) model_size = part->size;
    
    uint8_t* psram_buf = (uint8_t*)heap_caps_malloc(model_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (psram_buf != NULL) {
        if (+kibo_embedded_model_end > +kibo_embedded_model_start) {
            memcpy(psram_buf, kibo_embedded_model_start, model_size);
        } else if (part != NULL) {
            esp_partition_read(part, 0, psram_buf, model_size);
        }
        if (read_u32(psram_buf) == MODEL_MAGIC) {
            mmap_base = psram_buf;
            printf("[kibo-idf] psram model buffer loaded at 0x%08lX (%.2f MB)\n", (unsigned long)(uintptr_t)mmap_base, model_size / (1024.0 * 1024.0));
        }
    }
    
    if (!mmap_base) {
        printf("[kibo-idf] error: invalid model magic header\n");
        return false;
    }
    
    uint32_t vocab_size = read_u32(mmap_base + 4);
    uint32_t n_embd = read_u32(mmap_base + 8);
    uint32_t n_layer = read_u32(mmap_base + 16);
    uint32_t num_tensors = read_u32(mmap_base + 24);
    
    printf("[kibo-idf] tensor layout: vocab=%lu embd=%lu layers=%lu tensors=%lu\n", 
           (unsigned long)vocab_size, (unsigned long)n_embd, (unsigned long)n_layer, (unsigned long)num_tensors);
    
    uint32_t offset = sizeof(uint32_t) * 7;
    
    for (int t = 0; t < (int)num_tensors; t++) {
        int32_t name_len = read_i32(mmap_base + offset); offset += 4;
        if (name_len <= 0 || name_len >= 64) {
            printf("[kibo-idf] error: invalid tensor name length: %ld\n", (long)name_len);
            return false;
        }
        
        char tensor_name[64] = {0};
        safe_flash_copy(tensor_name, mmap_base + offset, name_len); offset += name_len;
        
        int32_t is_quant = read_i32(mmap_base + offset); offset += 4;
        float scale = read_f32(mmap_base + offset); offset += 4;
        int32_t num_elements = read_i32(mmap_base + offset); offset += 4;
        
        const uint8_t* data_ptr = mmap_base + offset;
        uint32_t data_bytes = (is_quant ? num_elements : num_elements * 4);
        
        std::string name(tensor_name);
        
        if (name == "tok_emb.weight") {
            kibo_model.tok_emb_scale = scale;
            kibo_model.tok_emb_w = (const int8_t*)data_ptr;
        } else if (name == "pos_emb.weight") {
            kibo_model.pos_emb_scale = scale;
            kibo_model.pos_emb_w = (const int8_t*)data_ptr;
        } else if (name == "head.weight") {
            kibo_model.head_scale = scale;
            kibo_model.head_w = (const int8_t*)data_ptr;
        } else if (name == "ln_f.weight") {
            safe_flash_copy(kibo_model.ln_f_w, data_ptr, num_elements * sizeof(float));
        } else if (name == "ln_f.bias") {
            safe_flash_copy(kibo_model.ln_f_b, data_ptr, num_elements * sizeof(float));
        } else if (name.rfind("blocks.", 0) == 0) {
            int layer = name[7] - '0';
            if (layer >= 0 && layer < N_LAYER) {
                TransformerBlock& blk = kibo_model.blocks[layer];
                if (name.find("ln_1.weight") != std::string::npos) safe_flash_copy(blk.ln1_w, data_ptr, num_elements * sizeof(float));
                else if (name.find("ln_1.bias") != std::string::npos) safe_flash_copy(blk.ln1_b, data_ptr, num_elements * sizeof(float));
                else if (name.find("attn.c_attn.weight") != std::string::npos) { blk.qkv_scale = scale; blk.qkv_w = (const int8_t*)data_ptr; }
                else if (name.find("attn.c_attn.bias") != std::string::npos) safe_flash_copy(blk.qkv_b, data_ptr, num_elements * sizeof(float));
                else if (name.find("attn.c_proj.weight") != std::string::npos) { blk.proj_scale = scale; blk.proj_w = (const int8_t*)data_ptr; }
                else if (name.find("attn.c_proj.bias") != std::string::npos) safe_flash_copy(blk.proj_b, data_ptr, num_elements * sizeof(float));
                else if (name.find("ln_2.weight") != std::string::npos) safe_flash_copy(blk.ln2_w, data_ptr, num_elements * sizeof(float));
                else if (name.find("ln_2.bias") != std::string::npos) safe_flash_copy(blk.ln2_b, data_ptr, num_elements * sizeof(float));
                else if (name.find("mlp.c_fc.weight") != std::string::npos) { blk.fc_scale = scale; blk.fc_w = (const int8_t*)data_ptr; }
                else if (name.find("mlp.c_fc.bias") != std::string::npos) safe_flash_copy(blk.fc_b, data_ptr, num_elements * sizeof(float));
                else if (name.find("mlp.c_proj.weight") != std::string::npos) { blk.mlp_proj_scale = scale; blk.mlp_proj_w = (const int8_t*)data_ptr; }
                else if (name.find("mlp.c_proj.bias") != std::string::npos) safe_flash_copy(blk.mlp_proj_b, data_ptr, num_elements * sizeof(float));
            }
        }
        
        offset += data_bytes;
    }
    
    printf("[kibo-idf] v3.0 native esp-idf dual-core engine ready!\n");
    return true;
}

void run_live_mcu_benchmark() {
    printf("\n==========================================================================\n");
    printf("  🔬 SCIENTIFIC HARDWARE BENCHMARK (ESP-IDF v5.x @ 240MHz XTENSA LX7)     \n");
    printf("==========================================================================\n");
    
    const int rows = 768;
    const int cols = 192;
    const int8_t* live_weights = kibo_model.blocks[0].fc_w;
    if (!live_weights) {
        printf("❌ Model weights not loaded!\n");
        return;
    }
    
    static float test_x[192];
    static float test_out[768];
    for (int i = 0; i < cols; i++) test_x[i] = 0.5f;
    
    const int iterations = 20;
    
    // Single-Core 8-Way Unrolled FPU
    int64_t t0 = esp_timer_get_time();
    for (int it = 0; it < iterations; it++) {
        matmul_int8_slice(test_out, test_x, live_weights, 0.005f, NULL, 0, rows, cols);
    }
    int64_t t1 = esp_timer_get_time();
    uint32_t single_core_us = (uint32_t)((t1 - t0) / iterations);
    
    // Dual-Core Parallel (Core 0+1)
    t0 = esp_timer_get_time();
    for (int it = 0; it < iterations; it++) {
        matmul_int8_vec(test_out, test_x, live_weights, 0.005f, NULL, rows, cols);
    }
    t1 = esp_timer_get_time();
    uint32_t dual_core_us = (uint32_t)((t1 - t0) / iterations);
    
    float speedup = (float)single_core_us / (float)(dual_core_us > 0 ? dual_core_us : 1);
    
    printf("  • Single-Core 8-Way Unrolled:    %6lu us\n", (unsigned long)single_core_us);
    printf("  • Dual-Core Parallel (Core 0+1): %6lu us (Speedup: %.2fx)\n", (unsigned long)dual_core_us, speedup);
    printf("==========================================================================\n\n");
}

void kibo_process_chat(const std::string& user_input) {
    std::string clean_cmd = user_input;
    while (!clean_cmd.empty() && (clean_cmd.back() == '\r' || clean_cmd.back() == '\n' || clean_cmd.back() == ' ')) {
        clean_cmd.pop_back();
    }
    
    if (clean_cmd == "benchmark" || clean_cmd == "test quant") {
        run_live_mcu_benchmark();
        printf("\nUser: ");
        fflush(stdout);
        return;
    }
    
    double math_res = 0;
    std::string math_op_str = "";
    if (kibo_math_calculator(clean_cmd, math_res, math_op_str)) {
        printf("\n[CALC: %s] [HAPPY] The result of %s is %.2f!\n\nUser: ", 
               math_op_str.c_str(), math_op_str.c_str(), math_res);
        fflush(stdout);
        return;
    }
    
    char prompt[256];
    snprintf(prompt, sizeof(prompt), "User: %s\nKibo:", clean_cmd.c_str());
    
    int tokens[MAX_SEQ_LEN];
    int n_tokens = tokenize_text(prompt, tokens, MAX_SEQ_LEN);
    if (n_tokens == 0) {
        printf("\nUser: ");
        fflush(stdout);
        return;
    }
    
    printf("\nKibo: ");
    fflush(stdout);
    
    int64_t t_start = esp_timer_get_time();
    int gen_tokens = 0;
    
    for (int p = 0; p < n_tokens - 1; p++) {
        forward_token(tokens[p], p, effective_max_seq_len);
    }
    
    int cur_token = tokens[n_tokens - 1];
    int pos = n_tokens - 1;
    
    while (pos < effective_max_seq_len - 1 && gen_tokens < 64) {
        forward_token(cur_token, pos, effective_max_seq_len);
        int next_token = sample_token(act_logits, VOCAB_SIZE, 0.7f);
        
        if (next_token == 0 || next_token == 1 || next_token == 3) {
            break;
        }
        
        const char* word = KIBO_VOCAB_TABLE[next_token];
        printf("%s", word);
        fflush(stdout);
        
        cur_token = next_token;
        pos++;
        gen_tokens++;
    }
    
    int64_t t_end = esp_timer_get_time();
    float elapsed_sec = (t_end - t_start) / 1000000.0f;
    float tok_per_sec = (elapsed_sec > 0.0f && gen_tokens > 0) ? ((float)gen_tokens / elapsed_sec) : 0.0f;
    
    printf("\n\n⚡ [%d tokens generated in %.2f s | Speed: %.1f tok/s]\n\nUser: ", 
           gen_tokens, elapsed_sec, tok_per_sec);
    fflush(stdout);
}
