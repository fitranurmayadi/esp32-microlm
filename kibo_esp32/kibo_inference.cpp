#include "kibo_inference.h"

KiboModel kibo_model;
KiboTelemetry kibo_telemetry = {0, 0.0f, 0.0f, 0, 0, 240, false};

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
#include "esp_idf_version.h"
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
static esp_partition_mmap_handle_t mmap_handle;
#define KIBO_MMAP_DATA ESP_PARTITION_MMAP_DATA
#else
#include "esp_spi_flash.h"
static spi_flash_mmap_handle_t mmap_handle;
#define KIBO_MMAP_DATA SPI_FLASH_MMAP_DATA
#endif

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

// Core 0 Dedicated FreeRTOS Worker Task
static void kibo_core0_worker_task(void* param) {
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
            kibo_core0_worker_task,
            "kibo_core0_worker",
            4096,
            NULL,
            5,
            &core0_task_handle,
            0 // Pin to Core 0 (PRO_CPU)
        );
        if (ret == pdPASS) {
            kibo_telemetry.dual_core_active = true;
            Serial.println("[kibo] dual-core parallel engine active (core 0 + core 1 @ 240MHz)");
        } else {
            kibo_telemetry.dual_core_active = false;
            Serial.println("[kibo] warning: running in single-core mode");
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
        
        yield();
    }
    
    layer_norm(act_xb, act_x, kibo_model.ln_f_w, kibo_model.ln_f_b, N_EMBD);
    matmul_int8_vec(act_logits, act_xb, kibo_model.head_w, kibo_model.head_scale, NULL, VOCAB_SIZE, N_EMBD);
    yield();
}

static int tokenize_text(const char* text, int* token_ids, int max_len) {
    int count = 0;
    int len = strlen(text);
    int i = 0;
    
    while (i < len && count < max_len) {
        bool matched = false;
        
        for (int s = 0; s < KIBO_NUM_SPECIAL_TOKENS; s++) {
            const char* st = KIBO_SPECIAL_TOKENS[s];
            int st_len = strlen(st);
            if (strncmp(&text[i], st, st_len) == 0) {
                for (int v = 0; v < VOCAB_SIZE; v++) {
                    if (strcmp(KIBO_VOCAB_TABLE[v], st) == 0) {
                        token_ids[count++] = v;
                        i += st_len;
                        matched = true;
                        break;
                    }
                }
                if (matched) break;
            }
        }
        if (matched) continue;
        
        char single_char[2] = { text[i], '\0' };
        int token_found = -1;
        for (int v = 0; v < VOCAB_SIZE; v++) {
            if (strcmp(KIBO_VOCAB_TABLE[v], single_char) == 0) {
                token_found = v;
                break;
            }
        }
        
        if (token_found != -1) {
            token_ids[count++] = token_found;
        } else {
            token_ids[count++] = 1; // <UNK>
        }
        i++;
    }
    return count;
}

// Deterministic Math Evaluation
static bool kibo_math_calculator(const String& input, String& response) {
    String clean = input;
    clean.toLowerCase();
    clean.replace("?", "");
    clean.replace("!", "");
    clean.replace(",", "");
    
    // English & Indonesian Normalization
    clean.replace(" dikali ", " * ");
    clean.replace(" kali ", " * ");
    clean.replace(" multiplied by ", " * ");
    clean.replace(" times ", " * ");
    clean.replace(" dibagi ", " / ");
    clean.replace(" bagi ", " / ");
    clean.replace(" divided by ", " / ");
    clean.replace(" over ", " / ");
    clean.replace(" ditambah ", " + ");
    clean.replace(" tambah ", " + ");
    clean.replace(" plus ", " + ");
    clean.replace(" added to ", " + ");
    clean.replace(" dikurang ", " - ");
    clean.replace(" dikurangi ", " - ");
    clean.replace(" kurang ", " - ");
    clean.replace(" minus ", " - ");
    clean.replace(" subtracted by ", " - ");
    clean.replace(" x ", " * ");
    clean.replace(" : ", " / ");
    clean.trim();

    double a = 0, b = 0;
    char op = 0;
    bool found = false;

    for (int i = 0; i < (int)clean.length(); i++) {
        char c = clean[i];
        if (c == '+' || c == '-' || c == '*' || c == '/') {
            String left_str = clean.substring(0, i);
            String right_str = clean.substring(i + 1);
            
            left_str.trim();
            right_str.trim();
            
            int last_space = left_str.lastIndexOf(' ');
            if (last_space != -1) {
                left_str = left_str.substring(last_space + 1);
            }
            int first_space = right_str.indexOf(' ');
            if (first_space != -1) {
                right_str = right_str.substring(0, first_space);
            }
            
            if (left_str.length() > 0 && right_str.length() > 0) {
                char* end1;
                char* end2;
                a = strtod(left_str.c_str(), &end1);
                b = strtod(right_str.c_str(), &end2);
                if (end1 != left_str.c_str() && end2 != right_str.c_str()) {
                    op = c;
                    found = true;
                    break;
                }
            }
        }
    }

    if (!found) return false;

    double res = 0.0;
    if (op == '+') res = a + b;
    else if (op == '-') res = a - b;
    else if (op == '*') res = a * b;
    else if (op == '/') {
        if (fabs(b) < 1e-9) {
            response = "[CALC: error] [ANGRY] Error: Division by zero is undefined!";
            return true;
        }
        res = a / b;
    }

    char res_buf[64];
    if (fabs(res - round(res)) < 1e-6) {
        snprintf(res_buf, sizeof(res_buf), "%.0f", res);
    } else {
        snprintf(res_buf, sizeof(res_buf), "%.4g", res);
    }

    char a_buf[32], b_buf[32];
    snprintf(a_buf, sizeof(a_buf), (fabs(a - round(a)) < 1e-6) ? "%.0f" : "%.4g", a);
    snprintf(b_buf, sizeof(b_buf), (fabs(b - round(b)) < 1e-6) ? "%.0f" : "%.4g", b);

    char out[128];
    snprintf(out, sizeof(out), "[CALC: %s %c %s = %s] [HAPPY] The result of %s %c %s is %s!",
             a_buf, op, b_buf, res_buf, a_buf, op, b_buf, res_buf);
    
    response = String(out);
    return true;
}

// Hardware & Robotic Tool Dispatcher
static bool kibo_hardware_action_dispatcher(const String& input, String& response) {
    String cmd = input;
    cmd.trim();
    cmd.toLowerCase();
    
    if (cmd == "status" || cmd == "system status" || cmd == "[cmd: status]") {
        char buf[384];
        snprintf(buf, sizeof(buf),
            "\n┌─── [SYSTEM STATUS & TELEMETRY] ──────────────────────────┐\n"
            "│ SoC: ESP32-S3 Dual-Core LX7 @ %d MHz                    │\n"
            "│ Engine: v2.0 Dual-Core SIMD Engine (%s)   │\n"
            "│ Free SRAM (Internal): %6u KB                           │\n"
            "│ Free PSRAM (Octal):   %6.2f MB                           │\n"
            "│ Total Tokens Generated: %-8u                         │\n"
            "│ Last Generation Speed:  %-4.1f tok/s                     │\n"
            "└──────────────────────────────────────────────────────────┘",
            ESP.getCpuFreqMHz(),
            kibo_telemetry.dual_core_active ? "Core 0+1 Active" : "Single-Core",
            ESP.getFreeHeap() / 1024,
            ESP.getFreePsram() / (1024.0f * 1024.0f),
            kibo_telemetry.total_tokens_generated,
            kibo_telemetry.last_tokens_per_sec
        );
        response = String(buf);
        return true;
    } else if (cmd.startsWith("eye ") || cmd.startsWith("[cmd: eye_")) {
        if (cmd.indexOf("happy") != -1) {
            response = "[ACTION: DISPLAY] Eyes -> HAPPY (^ _ ^)";
            return true;
        } else if (cmd.indexOf("sad") != -1) {
            response = "[ACTION: DISPLAY] Eyes -> SAD (T _ T)";
            return true;
        } else if (cmd.indexOf("blink") != -1) {
            response = "[ACTION: DISPLAY] Eyes -> BLINK (- _ -)";
            return true;
        }
    } else if (cmd.startsWith("servo ") || cmd.startsWith("[cmd: servo")) {
        int angle = 90;
        int idx = cmd.lastIndexOf(' ');
        if (idx != -1) {
            angle = cmd.substring(idx + 1).toInt();
        }
        char buf[128];
        snprintf(buf, sizeof(buf), "[ACTION: SERVO] Actuating robot servo to %d degrees", angle);
        response = String(buf);
        return true;
    }
    
    return false;
}

extern "C" {
    extern const uint8_t kibo_embedded_model_start[];
    extern const uint8_t kibo_embedded_model_end[];
}

bool kibo_init_model() {
    Serial.println("\n[kibo] initializing esp32 micro-lm v2.0...");
    
    // Strict SRAM Allocation for Working Activations
    act_x        = (float*)heap_caps_malloc(N_EMBD * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    act_xb       = (float*)heap_caps_malloc(N_EMBD * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    act_qkv      = (float*)heap_caps_malloc(3 * N_EMBD * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    act_att      = (float*)heap_caps_malloc(N_HEAD * MAX_SEQ_LEN * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    act_mlp      = (float*)heap_caps_malloc(4 * N_EMBD * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    act_proj     = (float*)heap_caps_malloc(N_EMBD * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    act_mlp_proj = (float*)heap_caps_malloc(N_EMBD * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    act_logits   = (float*)heap_caps_malloc(VOCAB_SIZE * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    
    // Octal PSRAM Allocation for 128-token KV-Cache
    if (psramFound() && ESP.getFreePsram() >= 1000000) {
        effective_max_seq_len = MAX_SEQ_LEN; // 128 context
        size_t kv_size = N_LAYER * effective_max_seq_len * N_EMBD * sizeof(float);
        kv_k_cache = (float*)heap_caps_malloc(kv_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        kv_v_cache = (float*)heap_caps_malloc(kv_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    } else {
        effective_max_seq_len = 42; // SRAM Fallback
        size_t kv_size = N_LAYER * effective_max_seq_len * N_EMBD * sizeof(float);
        kv_k_cache = (float*)malloc(kv_size);
        kv_v_cache = (float*)malloc(kv_size);
    }
    
    if (!act_x || !act_qkv || !kv_k_cache || !kv_v_cache) {
        Serial.println("error: failed to allocate working buffers");
        return false;
    }
    
    // Initialize Dual-Core Worker Task
    kibo_init_dual_core();

    const esp_partition_t* part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_DATA_FAT, NULL);
    if (!part) {
        part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "ffat");
    }
    
    size_t model_size = (kibo_embedded_model_end > kibo_embedded_model_start) ? (size_t)(kibo_embedded_model_end - kibo_embedded_model_start) : 2 * 1024 * 1024;
    if (part != NULL && part->size < model_size) {
        model_size = part->size;
    }
    if (psramFound() && ESP.getFreePsram() >= model_size) {
        uint8_t* psram_buf = (uint8_t*)heap_caps_malloc(model_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (psram_buf != NULL) {
            if (kibo_embedded_model_start != NULL && (kibo_embedded_model_end > kibo_embedded_model_start)) {
                memcpy(psram_buf, kibo_embedded_model_start, model_size);
            } else if (part != NULL) {
                esp_partition_read(part, 0, psram_buf, model_size);
            }
            if (read_u32(psram_buf) == MODEL_MAGIC) {
                mmap_base = psram_buf;
                Serial.printf("[kibo] psram buffer loaded at 0x%08X (%.2f MB)\n", (uint32_t)mmap_base, model_size / (1024.0 * 1024.0));
            }
        }
    }
    
    if (mmap_base == NULL && part != NULL) {
        size_t map_size = (part->size < 2 * 1024 * 1024) ? part->size : 2 * 1024 * 1024;
        const void* ptr = NULL;
        esp_err_t err = esp_partition_mmap(part, 0, map_size, KIBO_MMAP_DATA, &ptr, &mmap_handle);
        if (err == ESP_OK && ptr != NULL && read_u32((const uint8_t*)ptr) == MODEL_MAGIC) {
            mmap_base = (const uint8_t*)ptr;
            Serial.printf("[kibo] flash mmap mapped at 0x%08X\n", (uint32_t)mmap_base);
        }
    }
    
    if (mmap_base == NULL) {
        mmap_base = kibo_embedded_model_start;
        Serial.printf("[kibo] loaded from embedded flash at 0x%08X\n", (uint32_t)mmap_base);
    }
    
    uint32_t magic = read_u32(mmap_base);
    if (magic != MODEL_MAGIC) {
        Serial.printf("[kibo] error: invalid magic header (0x%08X != 0x%08X)\n", magic, MODEL_MAGIC);
        return false;
    }
    
    uint32_t vocab_size = read_u32(mmap_base + 4);
    uint32_t n_embd = read_u32(mmap_base + 8);
    uint32_t n_layer = read_u32(mmap_base + 16);
    uint32_t num_tensors = read_u32(mmap_base + 24);
    
    Serial.printf("[kibo] tensor layout: vocab=%d embd=%d layers=%d tensors=%d\n", vocab_size, n_embd, n_layer, num_tensors);
    
    uint32_t offset = sizeof(uint32_t) * 7;
    
    for (int t = 0; t < (int)num_tensors; t++) {
        int32_t name_len = read_i32(mmap_base + offset); offset += 4;
        if (name_len <= 0 || name_len >= 64) {
            Serial.printf("[kibo] error: invalid tensor name length: %d\n", name_len);
            return false;
        }
        
        char tensor_name[64] = {0};
        safe_flash_copy(tensor_name, mmap_base + offset, name_len); offset += name_len;
        
        int32_t is_quant = read_i32(mmap_base + offset); offset += 4;
        float scale = read_f32(mmap_base + offset); offset += 4;
        int32_t num_elements = read_i32(mmap_base + offset); offset += 4;
        
        const uint8_t* data_ptr = mmap_base + offset;
        uint32_t data_bytes = (is_quant ? num_elements : num_elements * 4);
        
        String name = String(tensor_name);
        
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
        } else if (name.startsWith("blocks.")) {
            int layer = name.substring(7, 8).toInt();
            if (layer >= 0 && layer < N_LAYER) {
                TransformerBlock& blk = kibo_model.blocks[layer];
                if (name.endsWith("ln_1.weight")) safe_flash_copy(blk.ln1_w, data_ptr, num_elements * sizeof(float));
                else if (name.endsWith("ln_1.bias")) safe_flash_copy(blk.ln1_b, data_ptr, num_elements * sizeof(float));
                else if (name.endsWith("attn.c_attn.weight")) { blk.qkv_scale = scale; blk.qkv_w = (const int8_t*)data_ptr; }
                else if (name.endsWith("attn.c_attn.bias")) safe_flash_copy(blk.qkv_b, data_ptr, num_elements * sizeof(float));
                else if (name.endsWith("attn.c_proj.weight")) { blk.proj_scale = scale; blk.proj_w = (const int8_t*)data_ptr; }
                else if (name.endsWith("attn.c_proj.bias")) safe_flash_copy(blk.proj_b, data_ptr, num_elements * sizeof(float));
                else if (name.endsWith("ln_2.weight")) safe_flash_copy(blk.ln2_w, data_ptr, num_elements * sizeof(float));
                else if (name.endsWith("ln_2.bias")) safe_flash_copy(blk.ln2_b, data_ptr, num_elements * sizeof(float));
                else if (name.endsWith("mlp.c_fc.weight")) { blk.fc_scale = scale; blk.fc_w = (const int8_t*)data_ptr; }
                else if (name.endsWith("mlp.c_fc.bias")) safe_flash_copy(blk.fc_b, data_ptr, num_elements * sizeof(float));
                else if (name.endsWith("mlp.c_proj.weight")) { blk.mlp_proj_scale = scale; blk.mlp_proj_w = (const int8_t*)data_ptr; }
                else if (name.endsWith("mlp.c_proj.bias")) safe_flash_copy(blk.mlp_proj_b, data_ptr, num_elements * sizeof(float));
            }
        }
        
        offset += data_bytes;
    }
    
    Serial.println("[kibo] v2.0 dual-core int8 engine ready");
    return true;
}

// Hardware Memory & Matmul Benchmark Suite
void run_live_mcu_benchmark() {
    Serial.println("\n==========================================================================");
    Serial.println("  🔬 SCIENTIFIC HARDWARE BENCHMARK (ESP32-S3 @ 240MHz XTENSA LX7)        ");
    Serial.println("==========================================================================");
    
    // 1. Memory Read Bandwidth Test
    Serial.println("\n[1] MEMORY READ BANDWIDTH BENCHMARK (10MB Sequential Scan):");
    const size_t test_buf_size = 64 * 1024; // 64 KB
    uint8_t* sram_buf = (uint8_t*)heap_caps_malloc(test_buf_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    uint8_t* psram_buf = (uint8_t*)heap_caps_malloc(test_buf_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    
    volatile uint32_t sum = 0;
    const int mem_iters = 160; // 160 * 64KB = 10.24 MB
    
    if (sram_buf) {
        memset(sram_buf, 0x55, test_buf_size);
        int64_t t0 = esp_timer_get_time();
        for (int it = 0; it < mem_iters; it++) {
            const uint32_t* p = (const uint32_t*)sram_buf;
            for (size_t i = 0; i < test_buf_size / 4; i++) sum += p[i];
        }
        int64_t t1 = esp_timer_get_time();
        float sram_mbps = (10.24f * 1000000.0f) / (float)(t1 - t0);
        Serial.printf("  • Internal SRAM Bandwidth: %6.1f MB/s\n", sram_mbps);
        free(sram_buf);
    }
    
    if (psram_buf) {
        memset(psram_buf, 0xAA, test_buf_size);
        int64_t t0 = esp_timer_get_time();
        for (int it = 0; it < mem_iters; it++) {
            const uint32_t* p = (const uint32_t*)psram_buf;
            for (size_t i = 0; i < test_buf_size / 4; i++) sum += p[i];
        }
        int64_t t1 = esp_timer_get_time();
        float psram_mbps = (10.24f * 1000000.0f) / (float)(t1 - t0);
        Serial.printf("  • Octal PSRAM Bandwidth:   %6.1f MB/s\n", psram_mbps);
        free(psram_buf);
    }
    
    // 2. Dual-Core vs Single-Core Matmul Latency
    Serial.println("\n[2] COMPUTE ENGINE LATENCY (192 -> 768 Projection, 147,456 weights):");
    const int rows = 768;
    const int cols = 192;
    const int8_t* live_weights = kibo_model.blocks[0].fc_w;
    if (!live_weights) {
        Serial.println("❌ Model weights not loaded!");
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
    
    Serial.printf("  • Single-Core 8-Way Unrolled:  %6u us\n", single_core_us);
    Serial.printf("  • Dual-Core Parallel (Core 0+1): %6u us (Speedup: %.2fx)\n", dual_core_us, speedup);
    Serial.println("==========================================================================\n");
}

void kibo_process_chat(const String& user_input) {
    String clean_cmd = user_input;
    clean_cmd.trim();
    clean_cmd.toLowerCase();
    
    if (clean_cmd == "benchmark" || clean_cmd == "test quant" || clean_cmd == "benchmark hardware") {
        run_live_mcu_benchmark();
        Serial.print("\nUser: ");
        return;
    }

    String hw_resp;
    if (kibo_hardware_action_dispatcher(user_input, hw_resp)) {
        Serial.println(hw_resp);
        Serial.print("\nUser: ");
        return;
    }

    String math_resp;
    if (kibo_math_calculator(user_input, math_resp)) {
        Serial.println(math_resp);
        Serial.print("\nUser: ");
        return;
    }
    
    String prompt_text = "User: " + user_input + "\nKibo:";
    
    int token_ids[MAX_SEQ_LEN];
    int prompt_len = tokenize_text(prompt_text.c_str(), token_ids, MAX_SEQ_LEN);
    
    Serial.print("Kibo: ");
    
    int current_seq_len = prompt_len;
    
    for (int pos = 0; pos < prompt_len; pos++) {
        forward_token(token_ids[pos], pos, prompt_len);
    }
    
    int64_t gen_start = esp_timer_get_time();
    int tokens_generated = 0;

    for (int gen = 0; gen < 80; gen++) {
        if (current_seq_len >= effective_max_seq_len - 1) break;
        
        int next_token = 0;
        float max_logit = -1e9f;
        for (int v = 0; v < VOCAB_SIZE; v++) {
            if (act_logits[v] > max_logit) {
                max_logit = act_logits[v];
                next_token = v;
            }
        }
        
        if (next_token == KIBO_EOS_ID || next_token == 0) break;
        
        const char* token_str = KIBO_VOCAB_TABLE[next_token];
        Serial.print(token_str);
        tokens_generated++;
        
        if (strcmp(token_str, "\n") == 0 && gen > 0) break;
        
        token_ids[current_seq_len] = next_token;
        forward_token(next_token, current_seq_len, current_seq_len + 1);
        current_seq_len++;
    }
    
    int64_t gen_end = esp_timer_get_time();
    float total_s = (float)(gen_end - gen_start) / 1000000.0f;
    float tps = (total_s > 0.0f && tokens_generated > 0) ? ((float)tokens_generated / total_s) : 0.0f;

    kibo_telemetry.total_tokens_generated += tokens_generated;
    kibo_telemetry.last_generation_time_s = total_s;
    kibo_telemetry.last_tokens_per_sec = tps;

    Serial.println("");
    if (tokens_generated > 0) {
        Serial.printf("⚡ [%d tokens generated in %.2f s | Speed: %.1f tok/s]\n", tokens_generated, total_s, tps);
    }
    Serial.print("\nUser: ");
}
