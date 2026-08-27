#include "kibo_inference.h"

KiboModel kibo_model;

static float* act_x = NULL;
static float* act_xb = NULL;
static float* act_qkv = NULL;
static float* act_att = NULL;
static float* act_mlp = NULL;
static float* act_proj = NULL;
static float* act_mlp_proj = NULL;
static float* act_logits = NULL;

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

static void layer_norm(float* out, const float* x, const float* w, const float* b, int dim) {
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) mean += x[i];
    mean /= dim;
    
    float var = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= dim;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    
    for (int i = 0; i < dim; i++) {
        out[i] = (x[i] - mean) * inv_std * w[i] + b[i];
    }
}

static void matmul_int8_vec(float* out, const float* x, const int8_t* w, float scale, const float* bias, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float dot = 0.0f;
        const int8_t* w_row = w + r * cols;
        for (int c = 0; c < cols; c++) {
            dot += x[c] * (float)w_row[c];
        }
        float sum = dot * scale;
        if (bias != NULL) {
            sum += bias[r];
        }
        out[r] = sum;
    }
}

static float gelu_act(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
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
                
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) {
                    dot += q_h[d] * k_h[d];
                }
                dot *= scale;
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
                for (int d = 0; d < HEAD_DIM; d++) {
                    head_out[d] += a_t * v_h[d];
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

static bool kibo_math_calculator(const String& input, String& response) {
    String clean = input;
    clean.toLowerCase();
    clean.replace("?", "");
    clean.replace("!", "");
    clean.replace(",", "");
    
    // Normalisasi kata-kata matematika Bahasa Indonesia
    clean.replace(" dikali ", " * ");
    clean.replace(" kali ", " * ");
    clean.replace(" dibagi ", " / ");
    clean.replace(" bagi ", " / ");
    clean.replace(" ditambah ", " + ");
    clean.replace(" tambah ", " + ");
    clean.replace(" dikurang ", " - ");
    clean.replace(" dikurangi ", " - ");
    clean.replace(" kurang ", " - ");
    clean.replace(" x ", " * ");
    clean.replace(" : ", " / ");
    clean.trim();

    double a = 0, b = 0;
    char op = 0;
    bool found = false;
    
    int n = sscanf(clean.c_str(), "berapa %lf %c %lf", &a, &op, &b);
    if (n == 3) found = true;
    if (!found) {
        n = sscanf(clean.c_str(), "hitung %lf %c %lf", &a, &op, &b);
        if (n == 3) found = true;
    }
    if (!found) {
        n = sscanf(clean.c_str(), "hasil dari %lf %c %lf", &a, &op, &b);
        if (n == 3) found = true;
    }
    if (!found) {
        n = sscanf(clean.c_str(), "%lf %c %lf", &a, &op, &b);
        if (n == 3 && (op == '+' || op == '-' || op == '*' || op == '/')) {
            found = true;
        }
    }
    
    // Pencarian regex-like untuk "<angka> <operator> <angka>"
    if (!found) {
        int op_idx = -1;
        char found_op = 0;
        for (int i = 0; i < (int)clean.length(); i++) {
            char c = clean[i];
            if (c == '+' || c == '-' || c == '*' || c == '/') {
                if (i > 0 && (clean[i-1] == ' ' || isDigit(clean[i-1]))) {
                    op_idx = i;
                    found_op = c;
                    break;
                }
            }
        }
        if (op_idx > 0) {
            String left = clean.substring(0, op_idx);
            String right = clean.substring(op_idx + 1);
            left.trim();
            right.trim();
            int last_space = left.lastIndexOf(' ');
            if (last_space != -1) left = left.substring(last_space + 1);
            int next_space = right.indexOf(' ');
            if (next_space != -1) right = right.substring(0, next_space);
            
            if (left.length() > 0 && right.length() > 0) {
                a = left.toDouble();
                b = right.toDouble();
                op = found_op;
                found = true;
            }
        }
    }
    
    if (found) {
        double res = 0;
        if (op == '+') res = a + b;
        else if (op == '-') res = a - b;
        else if (op == '*') res = a * b;
        else if (op == '/') {
            if (b == 0) {
                response = "🧮 [Tool CALC Error]\nKibo: [ANGRY] Maaf, pembagian dengan nol tidak dapat dilakukan!";
                return true;
            }
            res = a / b;
        } else {
            return false;
        }
        
        String res_str = (res == (long)res) ? String((long)res) : String(res, 2);
        String a_str = (a == (long)a) ? String((long)a) : String(a, 2);
        String b_str = (b == (long)b) ? String((long)b) : String(b, 2);
        
        response = "🧮 [Tool CALC: " + a_str + " " + String(op) + " " + b_str + " = " + res_str + "]\n";
        response += "Kibo: [CALC: " + a_str + " " + String(op) + " " + b_str + "] [HAPPY] Hasil dari " + a_str + " " + String(op) + " " + b_str + " adalah " + res_str + "! Kibo pintar berhitung kan! 🧠✨";
        return true;
    }
    return false;
}

extern "C" const uint8_t kibo_embedded_model_start[];
extern "C" const uint8_t kibo_embedded_model_end[];

bool kibo_init_model() {
    act_x = (float*)malloc(N_EMBD * sizeof(float));
    act_xb = (float*)malloc(N_EMBD * sizeof(float));
    act_qkv = (float*)malloc(3 * N_EMBD * sizeof(float));
    act_att = (float*)malloc(N_HEAD * MAX_SEQ_LEN * sizeof(float));
    act_mlp = (float*)malloc(4 * N_EMBD * sizeof(float));
    act_proj = (float*)malloc(N_EMBD * sizeof(float));
    act_mlp_proj = (float*)malloc(N_EMBD * sizeof(float));
    act_logits = (float*)malloc(VOCAB_SIZE * sizeof(float));
    
    // Dynamic KV Cache: 128 context on PSRAM boards, 42 context on SRAM-only boards (Nano ESP32)
    if (psramFound() && ESP.getFreePsram() >= 1000000) {
        effective_max_seq_len = MAX_SEQ_LEN; // 128
        size_t kv_size = N_LAYER * effective_max_seq_len * N_EMBD * sizeof(float);
        kv_k_cache = (float*)heap_caps_malloc(kv_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        kv_v_cache = (float*)heap_caps_malloc(kv_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    } else {
        effective_max_seq_len = 42; // Max safe context in 512KB SRAM
        size_t kv_size = N_LAYER * effective_max_seq_len * N_EMBD * sizeof(float);
        kv_k_cache = (float*)malloc(kv_size);
        kv_v_cache = (float*)malloc(kv_size);
    }
    
    if (!act_x || !act_qkv || !kv_k_cache || !kv_v_cache) {
        Serial.println("error: gagal mengalokasikan buffer");
        return false;
    }
    
    Serial.printf("[kibo] alokasi working buffer selesai (%u bytes free heap)\n", ESP.getFreeHeap());

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
    
    Serial.printf("[kibo] layout tensor: vocab=%d embd=%d layers=%d tensors=%d\n", vocab_size, n_embd, n_layer, num_tensors);
    
    uint32_t offset = sizeof(uint32_t) * 7;
    
    for (int t = 0; t < (int)num_tensors; t++) {
        int32_t name_len = read_i32(mmap_base + offset); offset += 4;
        if (name_len <= 0 || name_len >= 64) {
            Serial.printf("[kibo] error: panjang nama tensor tidak valid: %d\n", name_len);
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
    
    Serial.println("[kibo] engine inferensi int8 siap");
    return true;
}

void run_live_mcu_benchmark() {
    Serial.println("\n==========================================================================");
    Serial.println("  🔬 PENGUJIAN KUANTISASI REAL-TIME ON-CHIP (ESP32-S3 @ 240MHz XTENSA LX7)");
    Serial.println("==========================================================================");
    Serial.printf("Dimensi Layer   : 192 -> 768 (147.456 parameter per projeksi matriks)\n");
    Serial.printf("Menguji bobot model live di 8MB Octal PSRAM...\n\n");

    const int rows = 768;
    const int cols = 192;
    const int num_w = rows * cols;
    const int8_t* live_weights = kibo_model.blocks[0].fc_w;
    if (!live_weights) {
        Serial.println("❌ Bobot model belum dimuat ke RAM!");
        return;
    }

    static float test_x[192];
    static float test_out[768];
    for (int i = 0; i < cols; i++) test_x[i] = 0.5f;

    const int iterations = 10;
    volatile float dummy_sink = 0.0f;

    // 1. INT8 Direct Byte Access (147,456 weights)
    int64_t t0 = esp_timer_get_time();
    for (int it = 0; it < iterations; it++) {
        for (int r = 0; r < rows; r++) {
            float dot = 0.0f;
            const int8_t* w_row = live_weights + r * cols;
            for (int c = 0; c < cols; c++) {
                dot += test_x[c] * (float)w_row[c];
            }
            test_out[r] = dot * 0.005f;
            dummy_sink += test_out[r];
        }
    }
    int64_t t1 = esp_timer_get_time();
    uint32_t int8_us = (uint32_t)((t1 - t0) / iterations);

    // 2. INT4 Nibble Unpacking (Bit-shift + Masking + Sign Extension)
    const uint8_t* packed_w = (const uint8_t*)live_weights;
    t0 = esp_timer_get_time();
    for (int it = 0; it < iterations; it++) {
        for (int r = 0; r < rows; r++) {
            float dot = 0.0f;
            const uint8_t* w_row = packed_w + r * (cols / 2);
            for (int c = 0; c < cols / 2; c++) {
                uint8_t p = w_row[c];
                int8_t w0 = (int8_t)((p & 0x0F) >= 8 ? (p & 0x0F) - 16 : (p & 0x0F));
                int8_t w1 = (int8_t)((p >> 4) >= 8 ? (p >> 4) - 16 : (p >> 4));
                dot += test_x[c * 2] * (float)w0 + test_x[c * 2 + 1] * (float)w1;
            }
            test_out[r] = dot * 0.01f;
            dummy_sink += test_out[r];
        }
    }
    t1 = esp_timer_get_time();
    uint32_t int4_us = (uint32_t)((t1 - t0) / iterations);

    float ratio = (float)int4_us / (float)(int8_us > 0 ? int8_us : 1);

    Serial.println("--------------------------------------------------------------------------");
    Serial.printf("%-10s | %-16s | %-18s | %-16s\n", "Format", "Ukuran Bobot", "Latensi MatMul (us)", "Kecepatan Relatif");
    Serial.println("--------------------------------------------------------------------------");
    Serial.printf("%-10s | %-16s | %10u us       | %-16s\n", "INT8", "1.79 MB", int8_us, "1.00x (Optimal)");
    Serial.printf("%-10s | %-16s | %10u us       | %.2fx (%s)\n", "INT4", "0.88 MB", int4_us, ratio, (int4_us > int8_us ? "Lebih lambat (unpack)" : "Lebih cepat"));
    Serial.println("==========================================================================\n");
    Serial.printf("💡 Hasil Pengujian Nyata Hardware ESP32-S3:\n");
    if (int4_us > int8_us) {
        float pct_slower = ((float)(int4_us - int8_us) * 100.0f) / (float)int8_us;
        Serial.printf("- Eksekusi INT4 lebih lambat +%.1f%% dibanding INT8 karena beban siklus CPU untuk bit-masking (& 0x0F), bit-shifting (>> 4), dan sign extension.\n", pct_slower);
    } else {
        Serial.println("- INT8 dan INT4 memiliki kecepatan seimbang pada bus Octal PSRAM.\n");
    }
}

void kibo_process_chat(const String& user_input) {
    String clean_cmd = user_input;
    clean_cmd.trim();
    clean_cmd.toLowerCase();
    if (clean_cmd == "benchmark" || clean_cmd == "uji kuantisasi" || clean_cmd == "test quant") {
        run_live_mcu_benchmark();
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

    Serial.println("");
    if (tokens_generated > 0) {
        Serial.printf("⚡ [%d token dihasilkan dalam %.2f s | Kecepatan: %.1f tok/dtk]\n", tokens_generated, total_s, tps);
    }
    Serial.print("\nUser: ");
}
