import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = "/home/aiot/Projects/KIBO-MICROLM/testing_video/carousel_slides"
ASSETS_DIR = "/home/aiot/Projects/KIBO-MICROLM/testing_video/assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Slide Dimensions (LinkedIn 4:5 Portrait Carousel)
W, H = 1080, 1350

# Color Palette (Obsidian Dark Minimalist)
BG_COLOR = (11, 15, 23)        # #0B0F17
CARD_BG = (21, 29, 42)         # #151D2A
CARD_BORDER = (38, 53, 74)     # #26354A
ACCENT_CYAN = (0, 229, 255)    # #00E5FF
ACCENT_AMBER = (255, 184, 0)   # #FFB800
ACCENT_GREEN = (16, 185, 129)  # #10B981
TEXT_WHITE = (255, 255, 255)
TEXT_MUTED = (148, 163, 184)   # #94A3B8

# Fonts
FONT_REGULAR = "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf"
FONT_MONO = "/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf"
FONT_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

def get_font(size, bold=False, mono=False, cjk=False):
    try:
        if cjk and os.path.exists(FONT_CJK):
            return ImageFont.truetype(FONT_CJK, size)
        if mono and os.path.exists(FONT_MONO):
            return ImageFont.truetype(FONT_MONO, size)
        if bold and os.path.exists(FONT_BOLD):
            return ImageFont.truetype(FONT_BOLD, size)
        if os.path.exists(FONT_REGULAR):
            return ImageFont.truetype(FONT_REGULAR, size)
        return ImageFont.load_default()
    except:
        return ImageFont.load_default()

def draw_pill_badge(draw, x, y, text, font, color, bg_color):
    bbox = font.getbbox(text)
    w = bbox[2] - bbox[0] + 28
    h = bbox[3] - bbox[1] + 18
    draw.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=bg_color, outline=color, width=1)
    draw.text((x + 14, y + 8), text, font=font, fill=color)
    return w, h

def create_base_slide(badge_text="TINYML & ON-DEVICE AI", slide_num=1, total_slides=7):
    img = Image.new("RGB", (W, H), BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    # Top Badge
    badge_font = get_font(18, bold=True)
    draw_pill_badge(draw, 60, 60, badge_text, badge_font, ACCENT_CYAN, (0, 40, 60))
    
    # Slide Number
    num_font = get_font(20, bold=True)
    num_str = f"{slide_num:02d} / {total_slides:02d}"
    draw.text((W - 140, 68), num_str, font=num_font, fill=TEXT_MUTED)
    
    # Footer
    footer_font = get_font(18, bold=False)
    draw.text((60, H - 65), "Fitra Nurmayadi  |  ESP32 Micro-LM", font=footer_font, fill=TEXT_MUTED)
    
    if slide_num < total_slides:
        draw.text((W - 170, H - 65), "Swipe >>", font=get_font(18, bold=True), fill=ACCENT_CYAN)
    else:
        draw.text((W - 270, H - 65), "github.com/fitranurmayadi", font=get_font(18, bold=True), fill=ACCENT_GREEN)
        
    return img, draw

# Generate Dark Matplotlib Benchmark Chart
def generate_benchmark_chart():
    chart_path = os.path.join(ASSETS_DIR, "benchmark_chart_dark.png")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2), facecolor='#151D2A')
    
    formats = ['INT8 (Selected)', 'INT4 (Nibble)']
    latencies = [6557, 8264] # in microseconds
    sizes = [1.79, 0.88] # in MB
    colors = ['#00E5FF', '#FFB800']
    
    # Chart 1: Latency
    bars1 = ax1.bar(formats, latencies, color=colors, width=0.55, edgecolor='#38BDF8', linewidth=1.5)
    ax1.set_facecolor('#151D2A')
    ax1.set_title('MatMul Execution Time (Lower is Faster)', color='#FFFFFF', fontsize=12, pad=12, fontweight='bold')
    ax1.set_ylabel('Latency (microseconds)', color='#94A3B8', fontsize=10)
    ax1.tick_params(colors='#94A3B8', labelsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.2, color='#94A3B8')
    ax1.set_ylim(0, 10000)
    
    for bar, lat in zip(bars1, latencies):
        yval = bar.get_height()
        speed_str = "1.00x (Baseline)" if lat == 6557 else "+26% Slower"
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 250, f"{lat:,} us\n({speed_str})", ha='center', va='bottom', color='#FFFFFF', fontsize=9.5, fontweight='bold')
        
    # Chart 2: Model Size
    bars2 = ax2.bar(formats, sizes, color=['#10B981', '#F59E0B'], width=0.55, edgecolor='#34D399', linewidth=1.5)
    ax2.set_facecolor('#151D2A')
    ax2.set_title('PSRAM Model Payload (MB)', color='#FFFFFF', fontsize=12, pad=12, fontweight='bold')
    ax2.set_ylabel('Size in PSRAM (MB)', color='#94A3B8', fontsize=10)
    ax2.tick_params(colors='#94A3B8', labelsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.2, color='#94A3B8')
    ax2.set_ylim(0, 2.5)
    
    for bar, sz in zip(bars2, sizes):
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.08, f"{sz:.2f} MB", ha='center', va='bottom', color='#FFFFFF', fontsize=10, fontweight='bold')
        
    for spine in ax1.spines.values():
        spine.set_color('#26354A')
    for spine in ax2.spines.values():
        spine.set_color('#26354A')
        
    plt.tight_layout()
    plt.savefig(chart_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    return chart_path

# ==========================================
# SLIDE 1: COVER
# ==========================================
def render_slide_1():
    img, draw = create_base_slide("TINYML & EMBEDDED AI", 1)
    
    t_font = get_font(52, bold=True)
    draw.text((60, 140), "Running a 1.84M", font=t_font, fill=TEXT_WHITE)
    draw.text((60, 210), "Generative Micro-LM", font=t_font, fill=ACCENT_CYAN)
    draw.text((60, 280), "on the ESP32-S3", font=t_font, fill=TEXT_WHITE)
    
    sub_font = get_font(22, bold=False)
    draw.text((60, 365), "A 100% offline, bare-metal C++ Transformer running on", font=sub_font, fill=TEXT_MUTED)
    draw.text((60, 400), "a $5 microcontroller with 8MB Octal PSRAM at ~12-13 tok/s.", font=sub_font, fill=TEXT_MUTED)
    
    dev_path = os.path.join(ASSETS_DIR, "esp32s3_devkit.jpg")
    if os.path.exists(dev_path):
        hw_img = Image.open(dev_path).convert("RGB")
        hw_img = hw_img.resize((960, 540), Image.Resampling.LANCZOS)
        draw.rounded_rectangle([60, 460, 1020, 1170], radius=16, fill=CARD_BG, outline=CARD_BORDER, width=2)
        img.paste(hw_img, (60, 480))
        
        draw.rounded_rectangle([85, 1070, 995, 1145], radius=10, fill=(11, 15, 23), outline=ACCENT_AMBER, width=2)
        spec_font = get_font(20, bold=True, mono=True)
        draw.text((105, 1093), "ESP32-S3 DevKit | Xtensa 240MHz | 8MB PSRAM | INT8 (1.79 MB)", font=spec_font, fill=ACCENT_AMBER)
        
    img.save(os.path.join(OUTPUT_DIR, "slide_01.png"))
    print("Slide 1 rendered")

# ==========================================
# SLIDE 2: MOTIVATION & PERSONA
# ==========================================
def render_slide_2():
    img, draw = create_base_slide("THE MOTIVATION & PERSONA", 2)
    
    t_font = get_font(46, bold=True)
    draw.text((60, 140), "Brain for a Mini Desktop Robot", font=t_font, fill=TEXT_WHITE)
    
    # Card 1
    draw.rounded_rectangle([60, 225, 1020, 505], radius=16, fill=CARD_BG, outline=CARD_BORDER, width=2)
    draw_pill_badge(draw, 90, 255, "THE CURIOSITY", get_font(16, bold=True), ACCENT_AMBER, (50, 40, 0))
    draw.text((90, 315), "Can a microcontroller run an LLM locally?", font=get_font(26, bold=True), fill=TEXT_WHITE)
    body1 = (
        "I wanted to see if a language model could run directly on an ESP32\n"
        "to power a mini desktop companion robot.\n"
        "Zero cloud dependencies, zero subscription costs, 100% offline."
    )
    draw.text((90, 365), body1, font=get_font(21), fill=TEXT_MUTED)
    
    # Card 2
    draw.rounded_rectangle([60, 535, 1020, 825], radius=16, fill=CARD_BG, outline=CARD_BORDER, width=2)
    draw_pill_badge(draw, 90, 565, "THE PERSONA", get_font(16, bold=True), ACCENT_CYAN, (0, 45, 65))
    
    # Text with Kanji
    draw.text((90, 625), "Meet Kibo (", font=get_font(26, bold=True), fill=TEXT_WHITE)
    kibo_w = get_font(26, bold=True).getbbox("Meet Kibo (")[2]
    draw.text((90 + kibo_w, 623), "希望", font=get_font(24, cjk=True), fill=ACCENT_CYAN)
    kanji_w = get_font(24, cjk=True).getbbox("希望")[2]
    draw.text((90 + kibo_w + kanji_w + 5, 625), " - Hope)", font=get_font(26, bold=True), fill=TEXT_WHITE)
    
    body2 = (
        "Named after the Japanese word for 'Hope'.\n"
        "Curated for interactive dialogue with emotion tags ([HAPPY],\n"
        "[NEUTRAL], [ANGRY]) to drive LCD eye animations & motor gestures."
    )
    draw.text((90, 675), body2, font=get_font(21), fill=TEXT_MUTED)
    
    # Card 3
    draw.rounded_rectangle([60, 855, 1020, 1165], radius=16, fill=CARD_BG, outline=CARD_BORDER, width=2)
    draw_pill_badge(draw, 90, 885, "THE FRAMEWORK", get_font(16, bold=True), ACCENT_GREEN, (0, 45, 30))
    draw.text((90, 945), "ESP32 Micro-LM Architecture", font=get_font(26, bold=True), fill=TEXT_WHITE)
    body3 = (
        "A modular, lightweight C++ inference engine designed for embedded\n"
        "chips, proving on-device generative intelligence is practical on $5 silicon.\n"
        "Fully customizable for any robotics or IoT domain."
    )
    draw.text((90, 995), body3, font=get_font(21), fill=TEXT_MUTED)
    
    img.save(os.path.join(OUTPUT_DIR, "slide_02.png"))
    print("Slide 2 rendered")

# ==========================================
# SLIDE 3: ON-CHIP ARCHITECTURE
# ==========================================
def render_slide_3():
    img, draw = create_base_slide("ON-CHIP ARCHITECTURE", 3)
    
    t_font = get_font(46, bold=True)
    draw.text((60, 140), "ESP32-S3 Memory & Compute Map", font=t_font, fill=TEXT_WHITE)
    
    specs = [
        ("COMPUTE ENGINE", "Dual-Core Xtensa LX7 @ 240MHz", "• Pure bare-metal C++ execution (Zero framework overhead)\n• SIMD byte-aligned vector dot products\n• Deterministic execution timing without garbage collection", ACCENT_CYAN, (0, 45, 65)),
        ("MEMORY ALLOCATION", "8MB Octal PSRAM (80MHz OPI)", "• 1.79 MB INT8 Model Weights mapped directly in RAM\n• Dynamic 128-token Key-Value (KV) Cache\n• Zero-wait activation buffers for multi-layer attention", ACCENT_AMBER, (50, 40, 0)),
        ("MODEL SPECIFICATION", "1.84M Parameters (INT8)", "• 4 Transformer Blocks | 192 Hidden Dimensions | 4 Heads\n• 91 Vocabulary Tokens (Subwords + Control Tags)\n• 100% Offline with zero internet or cloud uplink", ACCENT_GREEN, (0, 45, 30)),
    ]
    
    y = 225
    for tag, title, body, col, bg_pill in specs:
        draw.rounded_rectangle([60, y, 1020, y + 290], radius=16, fill=CARD_BG, outline=col, width=2)
        draw_pill_badge(draw, 90, y + 22, tag, get_font(15, bold=True), col, bg_pill)
        draw.text((90, y + 70), title, font=get_font(24, bold=True), fill=TEXT_WHITE)
        draw.line([90, y + 110, 990, y + 110], fill=CARD_BORDER, width=1)
        draw.text((90, y + 130), body, font=get_font(19), fill=TEXT_MUTED)
        y += 315
        
    img.save(os.path.join(OUTPUT_DIR, "slide_03.png"))
    print("Slide 3 rendered")

# ==========================================
# SLIDE 4: THE 5-STEP PIPELINE
# ==========================================
def render_slide_4():
    img, draw = create_base_slide("ENGINEERING WORKFLOW", 4)
    
    t_font = get_font(46, bold=True)
    draw.text((60, 140), "From PyTorch to Bare-Metal C++", font=t_font, fill=TEXT_WHITE)
    
    steps = [
        ("01", "Curate Dataset & Emotion Tags", "Structured conversational dialogue with [HAPPY], [NEUTRAL] expression tokens."),
        ("02", "Train Model in PyTorch", "Trained a compact 4-Layer Causal Transformer (1.84M parameters, 192 dim)."),
        ("03", "Symmetric INT8 Quantization", "Compressed model payload from 7.02 MB -> 1.79 MB (100% FP32 logit fidelity)."),
        ("04", "Bare-Metal C++ Forward Pass", "Implemented pure C++ inference engine without TFLite Micro or ONNX runtimes."),
        ("05", "Hybrid Hardware ALU Tool Dispatch", "Math expressions ('100 x 7') routed directly to hardware ALU (<0.1 ms execution)."),
    ]
    
    y = 225
    for num, title, desc in steps:
        draw.rounded_rectangle([60, y, 1020, y + 165], radius=14, fill=CARD_BG, outline=CARD_BORDER, width=2)
        draw.rounded_rectangle([85, y + 25, 155, y + 95], radius=10, fill=(0, 45, 65), outline=ACCENT_CYAN, width=1)
        draw.text((98, y + 36), num, font=get_font(26, bold=True, mono=True), fill=ACCENT_CYAN)
        
        draw.text((180, y + 25), title, font=get_font(23, bold=True), fill=TEXT_WHITE)
        draw.text((180, y + 68), desc, font=get_font(18), fill=TEXT_MUTED)
        y += 185
        
    img.save(os.path.join(OUTPUT_DIR, "slide_04.png"))
    print("Slide 4 rendered")

# ==========================================
# SLIDE 5: INT8 vs INT4 BENCHMARK (WITH MATPLOTLIB)
# ==========================================
def render_slide_5():
    img, draw = create_base_slide("EMPIRICAL HARDWARE BENCHMARK", 5)
    
    t_font = get_font(46, bold=True)
    draw.text((60, 140), "The INT8 vs INT4 Paradox on MCU", font=t_font, fill=TEXT_WHITE)
    
    sub_font = get_font(21, bold=False)
    draw.text((60, 205), "Live on-chip MatMul benchmark on ESP32-S3 @ 240MHz (147,456 weights):", font=sub_font, fill=TEXT_MUTED)
    
    # Generate & Embed Matplotlib Chart
    chart_path = generate_benchmark_chart()
    if os.path.exists(chart_path):
        chart_img = Image.open(chart_path).convert("RGB")
        chart_img = chart_img.resize((960, 430), Image.Resampling.LANCZOS)
        draw.rounded_rectangle([60, 245, 1020, 695], radius=16, fill=CARD_BG, outline=CARD_BORDER, width=2)
        img.paste(chart_img, (60, 255))
        
    # Insight Callout Box
    draw.rounded_rectangle([60, 720, 1020, 1165], radius=16, fill=CARD_BG, outline=ACCENT_CYAN, width=2)
    draw_pill_badge(draw, 90, 745, "THE HARDWARE EXPLANATION", get_font(15, bold=True), ACCENT_CYAN, (0, 45, 65))
    
    draw.text((90, 800), "Why INT4 is +26% slower on Microcontrollers:", font=get_font(24, bold=True), fill=TEXT_WHITE)
    
    insight_text = (
        "• On microcontrollers without dedicated tensor units (NPU), INT4\n"
        "  requires manual CPU unpacking for every single nibble:\n"
        "    - Bit-masking (& 0x0F) and bit-shifting (>> 4)\n"
        "    - 4-bit two's complement sign-extension\n\n"
        "• The CPU cycles wasted on bit-manipulation completely cancel out\n"
        "  the memory bandwidth advantage.\n\n"
        "• Conclusion: Byte-aligned INT8 is 26% faster with 100% FP32 logit fidelity!"
    )
    draw.text((90, 850), insight_text, font=get_font(20), fill=TEXT_MUTED)
    
    img.save(os.path.join(OUTPUT_DIR, "slide_05.png"))
    print("Slide 5 rendered")

# ==========================================
# SLIDE 6: CROPPED TERMINAL LIVE PROOF
# ==========================================
def render_slide_6():
    img, draw = create_base_slide("LIVE TERMINAL INFERENCE", 6)
    
    t_font = get_font(44, bold=True)
    draw.text((60, 140), "Real-Time UART Generation (~12-13 tok/s)", font=t_font, fill=TEXT_WHITE)
    
    # Load and embed cropped terminal
    crop_path = "/home/aiot/.gemini/antigravity-ide/brain/b745120a-0087-4bb2-b22a-7e8180b591c4/scratch/crop_terminal_perfect.jpg"
    if os.path.exists(crop_path):
        term_img = Image.open(crop_path).convert("RGB")
        term_img = term_img.resize((960, 620), Image.Resampling.LANCZOS)
        draw.rounded_rectangle([55, 215, 1025, 855], radius=16, fill=CARD_BG, outline=ACCENT_CYAN, width=2)
        img.paste(term_img, (60, 225))
        
    metrics = [
        ("12.2 tok/s", "Real-Time Speed", "Physical chip verified", ACCENT_CYAN),
        ("1.79 MB", "PSRAM Footprint", "8MB Octal PSRAM", ACCENT_AMBER),
        ("< 0.1 ms", "ALU Math Tool", "Zero hallucinations", ACCENT_GREEN),
    ]
    
    x = 60
    for val, label, sub, col in metrics:
        draw.rounded_rectangle([x, 885, x + 300, 1165], radius=14, fill=CARD_BG, outline=CARD_BORDER, width=2)
        draw.text((x + 25, 915), val, font=get_font(34, bold=True, mono=True), fill=col)
        draw.text((x + 25, 975), label, font=get_font(21, bold=True), fill=TEXT_WHITE)
        draw.text((x + 25, 1020), sub, font=get_font(17), fill=TEXT_MUTED)
        x += 330
        
    img.save(os.path.join(OUTPUT_DIR, "slide_06.png"))
    print("Slide 6 rendered")

# ==========================================
# SLIDE 7: MULTI-BOARD & OPEN SOURCE
# ==========================================
def render_slide_7():
    img, draw = create_base_slide("MULTI-BOARD & OPEN SOURCE", 7)
    
    t_font = get_font(46, bold=True)
    draw.text((60, 140), "Validated on 3 Form Factors", font=t_font, fill=TEXT_WHITE)
    
    boards = [
        ("ESP32-S3 DevKit N16R8", "16MB Flash | 8MB Octal PSRAM", "esp32s3_devkit.jpg"),
        ("Arduino Nano ESP32", "16MB Flash | 8MB Octal PSRAM", "arduino_nano_esp32.jpg"),
        ("Seeed Studio XIAO S3", "8MB Flash | 8MB Octal PSRAM", "seeed_xiao_esp32s3.jpg"),
    ]
    
    y = 220
    for name, specs, photo_file in boards:
        draw.rounded_rectangle([60, y, 1020, y + 195], radius=16, fill=CARD_BG, outline=CARD_BORDER, width=2)
        
        photo_path = os.path.join(ASSETS_DIR, photo_file)
        if os.path.exists(photo_path):
            b_img = Image.open(photo_path).convert("RGB")
            b_img = b_img.resize((210, 155), Image.Resampling.LANCZOS)
            img.paste(b_img, (80, y + 20))
            
        draw.text((320, y + 35), name, font=get_font(27, bold=True), fill=TEXT_WHITE)
        draw.text((320, y + 80), specs, font=get_font(21), fill=ACCENT_CYAN)
        draw.text((320, y + 125), "Verified on Hardware", font=get_font(18, bold=True), fill=ACCENT_GREEN)
        y += 225
        
    # CTA Card
    draw.rounded_rectangle([60, 920, 1020, 1165], radius=16, fill=(0, 40, 60), outline=ACCENT_CYAN, width=2)
    draw.text((95, 945), "100% Open Source on GitHub (MIT License)", font=get_font(24, bold=True), fill=ACCENT_CYAN)
    draw.text((95, 990), "github.com/fitranurmayadi/esp32-microlm", font=get_font(24, bold=True, mono=True), fill=TEXT_WHITE)
    draw.text((95, 1050), "Easily customize the dataset & persona for your own robot!", font=get_font(20), fill=TEXT_MUTED)
    
    img.save(os.path.join(OUTPUT_DIR, "slide_07.png"))
    print("Slide 7 rendered")

# Run all renders
render_slide_1()
render_slide_2()
render_slide_3()
render_slide_4()
render_slide_5()
render_slide_6()
render_slide_7()

# Combine into multi-page PDF for LinkedIn Document carousel
slide_files = [os.path.join(OUTPUT_DIR, f"slide_{i:02d}.png") for i in range(1, 8)]
images = [Image.open(f).convert("RGB") for f in slide_files]
pdf_path = os.path.join(OUTPUT_DIR, "esp32_microlm_carousel.pdf")
images[0].save(pdf_path, save_all=True, append_images=images[1:], resolution=150.0)
print(f"All 7 slides and PDF generated at {pdf_path}")
