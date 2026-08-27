import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = "/home/aiot/Projects/KIBO-MICROLM/testing_video/carousel_slides"
ASSETS_DIR = "/home/aiot/Projects/KIBO-MICROLM/testing_video/assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Slide Dimensions (LinkedIn 4:5 Portrait Carousel)
W, H = 1080, 1350

# Modern Light Mode Editorial Palette
BG_CANVAS = (248, 250, 252)     # #F8FAFC Slate 50
CARD_BG = (255, 255, 255)       # #FFFFFF Pure White
CARD_BORDER = (226, 232, 240)   # #E2E8F0 Slate 200
CARD_BORDER_DARK = (203, 213, 225) # #CBD5E1 Slate 300

TEXT_PRIMARY = (15, 23, 42)     # #0F172A Slate 900 (Deep Charcoal)
TEXT_SECONDARY = (71, 85, 105)  # #475569 Slate 600
TEXT_MUTED = (100, 116, 139)    # #64748B Slate 500

# Accent Colors
BLUE_PRIMARY = (2, 132, 199)    # #0284C7 Sky 600
BLUE_BG = (224, 242, 254)       # #E0F2FE Sky 100
BLUE_BORDER = (186, 230, 253)   # #BAE6FD Sky 200

AMBER_PRIMARY = (217, 119, 6)   # #D97706 Amber 600
AMBER_BG = (254, 243, 199)      # #FEF3C7 Amber 100
AMBER_BORDER = (253, 230, 138)  # #FDE68A Amber 200

GREEN_PRIMARY = (5, 150, 105)   # #059669 Emerald 600
GREEN_BG = (209, 250, 229)      # #D1FAE5 Emerald 100
GREEN_BORDER = (167, 243, 208)  # #A7F3D0 Emerald 200

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

def draw_shadowed_card(draw, bbox, radius=16, fill=CARD_BG, outline=CARD_BORDER, width=1):
    x0, y0, x1, y1 = bbox
    draw.rounded_rectangle([x0, y0 + 3, x1, y1 + 3], radius=radius, fill=(235, 240, 246))
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill, outline=outline, width=int(width))

def draw_badge(draw, x, y, text, font, text_color, bg_color, border_color):
    bbox = font.getbbox(text)
    w = bbox[2] - bbox[0] + 24
    h = bbox[3] - bbox[1] + 16
    draw.rounded_rectangle([x, y, x + w, y + h], radius=6, fill=bg_color, outline=border_color, width=1)
    draw.text((x + 12, y + 7), text, font=font, fill=text_color)
    return w, h

def create_base_slide(badge_text="TINYML & EMBEDDED AI", slide_num=1, total_slides=7):
    img = Image.new("RGB", (W, H), BG_CANVAS)
    draw = ImageDraw.Draw(img)
    
    badge_font = get_font(16, bold=True)
    draw_badge(draw, 60, 55, badge_text, badge_font, BLUE_PRIMARY, BLUE_BG, BLUE_BORDER)
    
    num_font = get_font(18, bold=True)
    num_str = f"{slide_num:02d} / {total_slides:02d}"
    draw.text((W - 130, 62), num_str, font=num_font, fill=TEXT_MUTED)
    
    # Progress Bar
    bar_w = (W - 120)
    draw.line([60, H - 90, W - 60, H - 90], fill=CARD_BORDER, width=2)
    step_w = bar_w / total_slides
    draw.line([60, H - 90, 60 + int(step_w * slide_num), H - 90], fill=BLUE_PRIMARY, width=3)
    
    footer_font = get_font(17, bold=False)
    draw.text((60, H - 60), "Fitra Nurmayadi  |  ESP32 Micro-LM", font=footer_font, fill=TEXT_MUTED)
    
    if slide_num < total_slides:
        draw.text((W - 160, H - 60), "Swipe >>", font=get_font(17, bold=True), fill=BLUE_PRIMARY)
    else:
        draw.text((W - 270, H - 60), "github.com/fitranurmayadi", font=get_font(17, bold=True), fill=GREEN_PRIMARY)
        
    return img, draw

# ==========================================
# LIGHT MODE MATPLOTLIB CHART
# ==========================================
def generate_light_benchmark_chart():
    chart_path = os.path.join(ASSETS_DIR, "benchmark_chart_light.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.0), facecolor='#FFFFFF')
    
    formats = ['INT8 (Selected)', 'INT4 (Nibble)']
    latencies = [6557, 8264]
    sizes = [1.79, 0.88]
    
    bars1 = ax1.bar(formats, latencies, color=['#0284C7', '#D97706'], width=0.52, edgecolor='#E2E8F0', linewidth=1)
    ax1.set_facecolor('#F8FAFC')
    ax1.set_title('MatMul Latency (Lower is Faster)', color='#0F172A', fontsize=12, pad=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (microseconds)', color='#475569', fontsize=10)
    ax1.tick_params(colors='#475569', labelsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.5, color='#CBD5E1')
    ax1.set_ylim(0, 10500)
    
    for bar, lat in zip(bars1, latencies):
        yval = bar.get_height()
        speed_str = "1.00x (Baseline)" if lat == 6557 else "+26% Slower"
        color = '#0369A1' if lat == 6557 else '#B45309'
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 300, f"{lat:,} us\n({speed_str})", ha='center', va='bottom', color=color, fontsize=9.5, fontweight='bold')
        
    bars2 = ax2.bar(formats, sizes, color=['#059669', '#F59E0B'], width=0.52, edgecolor='#E2E8F0', linewidth=1)
    ax2.set_facecolor('#F8FAFC')
    ax2.set_title('Octal PSRAM Footprint (MB)', color='#0F172A', fontsize=12, pad=12, fontweight='bold')
    ax2.set_ylabel('Model Size (MB)', color='#475569', fontsize=10)
    ax2.tick_params(colors='#475569', labelsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.5, color='#CBD5E1')
    ax2.set_ylim(0, 2.4)
    
    for bar, sz in zip(bars2, sizes):
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.08, f"{sz:.2f} MB", ha='center', va='bottom', color='#0F172A', fontsize=10, fontweight='bold')
        
    for spine in ax1.spines.values():
        spine.set_color('#CBD5E1')
    for spine in ax2.spines.values():
        spine.set_color('#CBD5E1')
        
    plt.tight_layout()
    plt.savefig(chart_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    return chart_path

# ==========================================
# SLIDE 1: COVER
# ==========================================
def render_slide_1():
    img, draw = create_base_slide("TINYML & ON-DEVICE AI", 1)
    
    t_font = get_font(52, bold=True)
    draw.text((60, 130), "Running a 1.84M", font=t_font, fill=TEXT_PRIMARY)
    draw.text((60, 195), "Generative Micro-LM", font=t_font, fill=BLUE_PRIMARY)
    draw.text((60, 260), "on the ESP32-S3", font=t_font, fill=TEXT_PRIMARY)
    
    sub_font = get_font(21, bold=False)
    draw.text((60, 345), "A 100% offline, bare-metal C++ Transformer running on", font=sub_font, fill=TEXT_SECONDARY)
    draw.text((60, 375), "a $5 microcontroller with 8MB Octal PSRAM at ~12-13 tok/s.", font=sub_font, fill=TEXT_SECONDARY)
    
    dev_path = os.path.join(ASSETS_DIR, "esp32s3_devkit.jpg")
    if os.path.exists(dev_path):
        hw_img = Image.open(dev_path).convert("RGB")
        hw_img = hw_img.resize((960, 560), Image.Resampling.LANCZOS)
        draw_shadowed_card(draw, [60, 430, 1020, 1180], radius=16, fill=CARD_BG, outline=CARD_BORDER_DARK, width=2)
        img.paste(hw_img, (60, 445))
        
        draw.rounded_rectangle([90, 1075, 990, 1150], radius=10, fill=(15, 23, 42), outline=AMBER_PRIMARY, width=2)
        spec_font = get_font(19, bold=True, mono=True)
        draw.text((115, 1098), "ESP32-S3 DevKit | Xtensa 240MHz | 8MB PSRAM | INT8 (1.79 MB)", font=spec_font, fill=(255, 215, 0))
        
    img.save(os.path.join(OUTPUT_DIR, "slide_01.png"))
    print("Slide 1 rendered")

# ==========================================
# SLIDE 2: THE MOTIVATION & PERSONA
# ==========================================
def render_slide_2():
    img, draw = create_base_slide("THE MOTIVATION & PERSONA", 2)
    
    t_font = get_font(46, bold=True)
    draw.text((60, 130), "Brain for a Mini Desktop Robot", font=t_font, fill=TEXT_PRIMARY)
    
    # Card 1: The Curiosity
    draw_shadowed_card(draw, [60, 215, 1020, 495], radius=16, fill=CARD_BG, outline=CARD_BORDER, width=1)
    draw_badge(draw, 90, 245, "THE CURIOSITY", get_font(15, bold=True), AMBER_PRIMARY, AMBER_BG, AMBER_BORDER)
    draw.text((90, 305), "Can an ESP32 run a language model locally?", font=get_font(25, bold=True), fill=TEXT_PRIMARY)
    body1 = (
        "I wanted to see if a language model could run directly on an ESP32\n"
        "to power a mini desktop companion robot.\n"
        "Zero cloud API dependencies, zero subscription costs, 100% offline."
    )
    draw.text((90, 355), body1, font=get_font(20), fill=TEXT_SECONDARY)
    
    # Card 2: The Persona Kibo
    draw_shadowed_card(draw, [60, 525, 1020, 815], radius=16, fill=CARD_BG, outline=CARD_BORDER, width=1)
    draw_badge(draw, 90, 555, "THE PERSONA", get_font(15, bold=True), BLUE_PRIMARY, BLUE_BG, BLUE_BORDER)
    
    draw.text((90, 615), "Meet Kibo (", font=get_font(25, bold=True), fill=TEXT_PRIMARY)
    kibo_w = get_font(25, bold=True).getbbox("Meet Kibo (")[2]
    draw.text((90 + kibo_w, 613), "希望", font=get_font(24, cjk=True), fill=BLUE_PRIMARY)
    kanji_w = get_font(24, cjk=True).getbbox("希望")[2]
    draw.text((90 + kibo_w + kanji_w + 5, 615), " - Hope)", font=get_font(25, bold=True), fill=TEXT_PRIMARY)
    
    body2 = (
        "Named after the Japanese word for 'Hope'.\n"
        "Designed for interactive dialogue with emotion tags ([HAPPY],\n"
        "[NEUTRAL], [ANGRY]) to drive LCD eye animations & motor gestures."
    )
    draw.text((90, 665), body2, font=get_font(20), fill=TEXT_SECONDARY)
    
    # Card 3: The Framework
    draw_shadowed_card(draw, [60, 845, 1020, 1155], radius=16, fill=CARD_BG, outline=CARD_BORDER, width=1)
    draw_badge(draw, 90, 875, "THE FRAMEWORK", get_font(15, bold=True), GREEN_PRIMARY, GREEN_BG, GREEN_BORDER)
    draw.text((90, 935), "ESP32 Micro-LM Engine", font=get_font(25, bold=True), fill=TEXT_PRIMARY)
    body3 = (
        "A lightweight, modular, and fully customizable C++ inference engine\n"
        "that proves on-chip generative intelligence is practical on $5 silicon.\n"
        "Easily customizable for any robotics or IoT domain."
    )
    draw.text((90, 985), body3, font=get_font(20), fill=TEXT_SECONDARY)
    
    img.save(os.path.join(OUTPUT_DIR, "slide_02.png"))
    print("Slide 2 rendered")

# ==========================================
# SLIDE 3: ON-CHIP ARCHITECTURE
# ==========================================
def render_slide_3():
    img, draw = create_base_slide("ON-CHIP ARCHITECTURE", 3)
    
    t_font = get_font(46, bold=True)
    draw.text((60, 130), "ESP32-S3 Compute & Memory Layout", font=t_font, fill=TEXT_PRIMARY)
    
    cards = [
        ("COMPUTE CORE", "Dual-Core Xtensa LX7 @ 240MHz", "• Pure bare-metal C++ execution (Zero framework overhead)\n• Direct SIMD-aligned vector dot products\n• Deterministic execution timing without garbage collection", BLUE_PRIMARY, BLUE_BG, BLUE_BORDER),
        ("MEMORY ALLOCATION", "8MB Octal PSRAM (80MHz OPI)", "• 1.79 MB INT8 Model Weights mapped directly in RAM\n• Dynamic 128-token Key-Value (KV) Cache\n• Zero-wait activation working buffers for multi-layer attention", AMBER_PRIMARY, AMBER_BG, AMBER_BORDER),
        ("MODEL PAYLOAD", "1.84M Parameters (INT8)", "• 4 Transformer Layers | 192 Hidden Dimensions | 4 Heads\n• 91 Vocabulary Tokens (Subwords + Control Tags)\n• 100% Offline with zero cloud dependencies", GREEN_PRIMARY, GREEN_BG, GREEN_BORDER),
    ]
    
    y = 215
    for tag, title, body, col, bg_col, b_col in cards:
        draw_shadowed_card(draw, [60, y, 1020, y + 295], radius=16, fill=CARD_BG, outline=CARD_BORDER, width=1)
        draw_badge(draw, 90, y + 22, tag, get_font(15, bold=True), col, bg_col, b_col)
        draw.text((90, y + 70), title, font=get_font(24, bold=True), fill=TEXT_PRIMARY)
        draw.line([90, y + 110, 990, y + 110], fill=CARD_BORDER, width=1)
        draw.text((90, y + 130), body, font=get_font(19), fill=TEXT_SECONDARY)
        y += 320
        
    img.save(os.path.join(OUTPUT_DIR, "slide_03.png"))
    print("Slide 3 rendered")

# ==========================================
# SLIDE 4: THE 5-STEP PIPELINE
# ==========================================
def render_slide_4():
    img, draw = create_base_slide("ENGINEERING WORKFLOW", 4)
    
    t_font = get_font(46, bold=True)
    draw.text((60, 130), "From PyTorch to Bare-Metal C++", font=t_font, fill=TEXT_PRIMARY)
    
    steps = [
        ("01", "Curate Dataset & Emotion Tags", "Structured conversational dialogue with [HAPPY], [NEUTRAL] expression tokens."),
        ("02", "Train Model in PyTorch", "Trained a compact 4-Layer Causal Transformer (1.84M parameters, 192 dim)."),
        ("03", "Symmetric INT8 Quantization", "Compressed weights from 7.02 MB -> 1.79 MB (100% FP32 logit fidelity)."),
        ("04", "Bare-Metal C++ Forward Pass", "Implemented pure C++ Transformer without TFLite Micro or ONNX runtimes."),
        ("05", "Hybrid Hardware ALU Tool Dispatch", "Math queries ('100 x 7') routed directly to hardware ALU (<0.1 ms execution)."),
    ]
    
    y = 215
    for num, title, desc in steps:
        draw_shadowed_card(draw, [60, y, 1020, y + 170], radius=14, fill=CARD_BG, outline=CARD_BORDER, width=1)
        # Left accent color strip
        draw.rounded_rectangle([60, y, 70, y + 170], radius=4, fill=BLUE_PRIMARY)
        
        # Number badge
        draw.rounded_rectangle([95, y + 25, 165, y + 95], radius=10, fill=BLUE_BG, outline=BLUE_BORDER, width=1)
        draw.text((108, y + 36), num, font=get_font(25, bold=True, mono=True), fill=BLUE_PRIMARY)
        
        draw.text((190, y + 25), title, font=get_font(23, bold=True), fill=TEXT_PRIMARY)
        draw.text((190, y + 70), desc, font=get_font(18), fill=TEXT_SECONDARY)
        y += 190
        
    img.save(os.path.join(OUTPUT_DIR, "slide_04.png"))
    print("Slide 4 rendered")

# ==========================================
# SLIDE 5: INT8 vs INT4 BENCHMARK (LIGHT MATPLOTLIB)
# ==========================================
def render_slide_5():
    img, draw = create_base_slide("EMPIRICAL HARDWARE BENCHMARK", 5)
    
    t_font = get_font(46, bold=True)
    draw.text((60, 130), "The INT8 vs INT4 Paradox on MCU", font=t_font, fill=TEXT_PRIMARY)
    
    sub_font = get_font(20, bold=False)
    draw.text((60, 195), "Live on-chip MatMul benchmark on ESP32-S3 @ 240MHz (147,456 weights):", font=sub_font, fill=TEXT_SECONDARY)
    
    chart_path = generate_light_benchmark_chart()
    if os.path.exists(chart_path):
        chart_img = Image.open(chart_path).convert("RGB")
        chart_img = chart_img.resize((960, 420), Image.Resampling.LANCZOS)
        draw_shadowed_card(draw, [60, 235, 1020, 675], radius=16, fill=CARD_BG, outline=CARD_BORDER_DARK, width=1)
        img.paste(chart_img, (60, 245))
        
    # Technical Insight Box
    draw_shadowed_card(draw, [60, 700, 1020, 1165], radius=16, fill=CARD_BG, outline=BLUE_BORDER, width=2)
    draw_badge(draw, 90, 725, "THE HARDWARE EXPLANATION", get_font(15, bold=True), BLUE_PRIMARY, BLUE_BG, BLUE_BORDER)
    
    draw.text((90, 780), "Why INT4 is +26% slower on Microcontrollers:", font=get_font(23, bold=True), fill=TEXT_PRIMARY)
    
    insight_text = (
        "• On microcontrollers without dedicated tensor units (NPU), INT4\n"
        "  requires manual CPU unpacking for every single nibble:\n"
        "    - Bit-masking (& 0x0F) and bit-shifting (>> 4)\n"
        "    - 4-bit two's complement sign-extension\n\n"
        "• The CPU cycles wasted on bit-manipulation completely cancel out\n"
        "  the memory bandwidth advantage.\n\n"
        "• Conclusion: Byte-aligned INT8 is 26% faster with 100% FP32 logit fidelity!"
    )
    draw.text((90, 830), insight_text, font=get_font(20), fill=TEXT_SECONDARY)
    
    img.save(os.path.join(OUTPUT_DIR, "slide_05.png"))
    print("Slide 5 rendered")

# ==========================================
# SLIDE 6: CROPPED TERMINAL IN LIGHT WINDOW CONTAINER
# ==========================================
def render_slide_6():
    img, draw = create_base_slide("LIVE TERMINAL INFERENCE", 6)
    
    t_font = get_font(44, bold=True)
    draw.text((60, 130), "Real-Time UART Generation (~12-13 tok/s)", font=t_font, fill=TEXT_PRIMARY)
    
    # Terminal Window Container
    draw_shadowed_card(draw, [55, 200, 1025, 860], radius=16, fill=(15, 23, 42), outline=CARD_BORDER_DARK, width=2)
    
    # Window Header
    draw.rounded_rectangle([55, 200, 1025, 245], radius=16, fill=(30, 41, 59))
    draw.ellipse([75, 216, 91, 232], fill=(239, 68, 68))   # Red
    draw.ellipse([99, 216, 115, 232], fill=(245, 158, 11)) # Yellow
    draw.ellipse([123, 216, 139, 232], fill=(16, 185, 129))# Green
    draw.text((440, 215), "serial_chat.py - 115200 baud", font=get_font(16, bold=True, mono=True), fill=(148, 163, 184))
    
    crop_path = "/home/aiot/.gemini/antigravity-ide/brain/b745120a-0087-4bb2-b22a-7e8180b591c4/scratch/crop_terminal_perfect.jpg"
    if os.path.exists(crop_path):
        term_img = Image.open(crop_path).convert("RGB")
        term_img = term_img.resize((960, 600), Image.Resampling.LANCZOS)
        img.paste(term_img, (60, 250))
        
    metrics = [
        ("12.2 tok/s", "Real-Time Speed", "Physical chip verified", BLUE_PRIMARY, BLUE_BG, BLUE_BORDER),
        ("1.79 MB", "PSRAM Footprint", "8MB Octal PSRAM", AMBER_PRIMARY, AMBER_BG, AMBER_BORDER),
        ("< 0.1 ms", "ALU Math Tool", "Zero hallucinations", GREEN_PRIMARY, GREEN_BG, GREEN_BORDER),
    ]
    
    x = 60
    for val, label, sub, col, bg_col, b_col in metrics:
        draw_shadowed_card(draw, [x, 885, x + 300, 1165], radius=14, fill=CARD_BG, outline=b_col, width=1)
        draw.text((x + 25, 915), val, font=get_font(32, bold=True, mono=True), fill=col)
        draw.text((x + 25, 975), label, font=get_font(21, bold=True), fill=TEXT_PRIMARY)
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
    draw.text((60, 130), "Validated on 3 Form Factors", font=t_font, fill=TEXT_PRIMARY)
    
    boards = [
        ("ESP32-S3 DevKit N16R8", "16MB Flash | 8MB Octal PSRAM", "esp32s3_devkit.jpg"),
        ("Arduino Nano ESP32", "16MB Flash | 8MB Octal PSRAM", "arduino_nano_esp32.jpg"),
        ("Seeed Studio XIAO S3", "8MB Flash | 8MB Octal PSRAM", "seeed_xiao_esp32s3.jpg"),
    ]
    
    y = 215
    for name, specs, photo_file in boards:
        draw_shadowed_card(draw, [60, y, 1020, y + 195], radius=16, fill=CARD_BG, outline=CARD_BORDER, width=1)
        
        photo_path = os.path.join(ASSETS_DIR, photo_file)
        if os.path.exists(photo_path):
            b_img = Image.open(photo_path).convert("RGB")
            b_img = b_img.resize((210, 155), Image.Resampling.LANCZOS)
            img.paste(b_img, (80, y + 20))
            
        draw.text((320, y + 35), name, font=get_font(26, bold=True), fill=TEXT_PRIMARY)
        draw.text((320, y + 80), specs, font=get_font(20), fill=BLUE_PRIMARY)
        draw.text((320, y + 125), "✓ 100% Tested & Verified on Hardware", font=get_font(18, bold=True), fill=GREEN_PRIMARY)
        y += 225
        
    # CTA Card (Clean without emoji box)
    draw_shadowed_card(draw, [60, 915, 1020, 1165], radius=16, fill=BLUE_BG, outline=BLUE_BORDER, width=2)
    draw.text((95, 940), "100% Open Source on GitHub (MIT License)", font=get_font(24, bold=True), fill=BLUE_PRIMARY)
    draw.text((95, 985), "github.com/fitranurmayadi/esp32-microlm", font=get_font(24, bold=True, mono=True), fill=TEXT_PRIMARY)
    draw.text((95, 1045), "Clone, train your custom dataset, and flash to your ESP32!", font=get_font(19), fill=TEXT_SECONDARY)
    
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

# Combine into multi-page PDF
slide_files = [os.path.join(OUTPUT_DIR, f"slide_{i:02d}.png") for i in range(1, 8)]
images = [Image.open(f).convert("RGB") for f in slide_files]
pdf_path = os.path.join(OUTPUT_DIR, "esp32_microlm_carousel_light.pdf")
images[0].save(pdf_path, save_all=True, append_images=images[1:], resolution=150.0)
print(f"All 7 Light Mode slides and PDF generated at {pdf_path}")
