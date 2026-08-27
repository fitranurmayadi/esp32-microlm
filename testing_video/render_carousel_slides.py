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
BG_CANVAS = (248, 250, 252)     # #F8FAFC
CARD_BG = (255, 255, 255)       # #FFFFFF
CARD_BORDER = (226, 232, 240)   # #E2E8F0
CARD_BORDER_DARK = (203, 213, 225) # #CBD5E1

TEXT_PRIMARY = (15, 23, 42)     # #0F172A (Deep Slate)
TEXT_SECONDARY = (71, 85, 105)  # #475569
TEXT_MUTED = (100, 116, 139)    # #64748B

# Accents
BLUE_PRIMARY = (2, 132, 199)    # #0284C7
BLUE_BG = (224, 242, 254)       # #E0F2FE
BLUE_BORDER = (186, 230, 253)   # #BAE6FD

AMBER_PRIMARY = (217, 119, 6)   # #D97706
AMBER_BG = (254, 243, 199)      # #FEF3C7
AMBER_BORDER = (253, 230, 138)  # #FDE68A

GREEN_PRIMARY = (5, 150, 105)   # #059669
GREEN_BG = (209, 250, 229)      # #D1FAE5
GREEN_BORDER = (167, 243, 208)  # #A7F3D0

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
# SLIDE 1: COVER
# ==========================================
def render_slide_1():
    img, draw = create_base_slide("TINYML & ON-DEVICE AI", 1)
    
    t_font = get_font(54, bold=True)
    draw.text((60, 130), "Running a 1.84M", font=t_font, fill=TEXT_PRIMARY)
    draw.text((60, 200), "Generative Micro-LM", font=t_font, fill=BLUE_PRIMARY)
    draw.text((60, 270), "on the ESP32-S3", font=t_font, fill=TEXT_PRIMARY)
    
    sub_font = get_font(23, bold=False)
    draw.text((60, 360), "100% Offline  •  Bare-Metal C++  •  ~12-13 tokens/sec", font=sub_font, fill=TEXT_SECONDARY)
    
    draw_shadowed_card(draw, [60, 420, 1020, 1180], radius=16, fill=CARD_BG, outline=CARD_BORDER_DARK, width=1)
    
    dev_path = os.path.join(ASSETS_DIR, "esp32s3_devkit_tight.jpg")
    if os.path.exists(dev_path):
        hw_img = Image.open(dev_path).convert("RGB")
        hw_img = hw_img.resize((275, 628), Image.Resampling.LANCZOS)
        img.paste(hw_img, (110, 485))
        
        specs_side = [
            ("N16R8 MODULE", "ESP32-S3-WROOM-1", "16MB Flash | 8MB Octal PSRAM", BLUE_PRIMARY, BLUE_BG, BLUE_BORDER),
            ("COMPUTE ENGINE", "Dual-Core Xtensa LX7", "Clocked @ 240MHz (Bare-metal C++)", AMBER_PRIMARY, AMBER_BG, AMBER_BORDER),
            ("ON-CHIP MODEL", "1.84M Transformer", "INT8 Quantized (1.79 MB in PSRAM)", GREEN_PRIMARY, GREEN_BG, GREEN_BORDER),
            ("INFERENCE SPEED", "12 to 13 tokens/sec", "~80 ms per token real-time generation", TEXT_PRIMARY, (241, 245, 249), CARD_BORDER),
        ]
        
        sy = 485
        for tag, title, desc, col, bg_col, b_col in specs_side:
            draw.rounded_rectangle([430, sy, 980, sy + 138], radius=12, fill=bg_col, outline=b_col, width=1)
            draw.text((455, sy + 18), tag, font=get_font(15, bold=True), fill=col)
            draw.text((455, sy + 48), title, font=get_font(23, bold=True), fill=TEXT_PRIMARY)
            draw.text((455, sy + 88), desc, font=get_font(18), fill=TEXT_SECONDARY)
            sy += 162
            
    img.save(os.path.join(OUTPUT_DIR, "slide_01.png"))
    print("Slide 1 rendered")

# ==========================================
# SLIDE 2: THE MOTIVATION (CLEAN & CONCISE)
# ==========================================
def render_slide_2():
    img, draw = create_base_slide("THE MOTIVATION & PERSONA", 2)
    
    t_font = get_font(48, bold=True)
    draw.text((60, 130), "Brain for a Mini Desktop Robot", font=t_font, fill=TEXT_PRIMARY)
    
    cards = [
        ("THE CURIOSITY", "Can an ESP32 think locally?", "Building on-device conversational intelligence without cloud APIs or monthly subscription costs.", AMBER_PRIMARY, AMBER_BG, AMBER_BORDER),
        ("THE PERSONA", "Meet Kibo (希望 - Hope)", "Interactive conversational robot persona running completely on-device.", BLUE_PRIMARY, BLUE_BG, BLUE_BORDER),
        ("THE FRAMEWORK", "ESP32 Micro-LM Engine", "A lightweight, modular, bare-metal C++ inference engine designed for microcontrollers.", GREEN_PRIMARY, GREEN_BG, GREEN_BORDER),
    ]
    
    y = 220
    for tag, title, body, col, bg_col, b_col in cards:
        draw_shadowed_card(draw, [60, y, 1020, y + 260], radius=16, fill=CARD_BG, outline=CARD_BORDER, width=1)
        draw_badge(draw, 90, y + 25, tag, get_font(15, bold=True), col, bg_col, b_col)
        
        if "希望" in title:
            draw.text((90, y + 78), "Meet Kibo (", font=get_font(28, bold=True), fill=TEXT_PRIMARY)
            kw = get_font(28, bold=True).getbbox("Meet Kibo (")[2]
            draw.text((90 + kw, y + 75), "希望", font=get_font(26, cjk=True), fill=BLUE_PRIMARY)
            kjw = get_font(26, cjk=True).getbbox("希望")[2]
            draw.text((90 + kw + kjw + 5, y + 78), " - Hope)", font=get_font(28, bold=True), fill=TEXT_PRIMARY)
        else:
            draw.text((90, y + 78), title, font=get_font(28, bold=True), fill=TEXT_PRIMARY)
            
        draw.text((90, y + 135), body, font=get_font(21), fill=TEXT_SECONDARY)
        y += 300
        
    img.save(os.path.join(OUTPUT_DIR, "slide_02.png"))
    print("Slide 2 rendered")

# ==========================================
# SLIDE 3: ON-CHIP ARCHITECTURE
# ==========================================
def render_slide_3():
    img, draw = create_base_slide("ON-CHIP ARCHITECTURE", 3)
    
    t_font = get_font(48, bold=True)
    draw.text((60, 130), "ESP32-S3 Hardware Limits", font=t_font, fill=TEXT_PRIMARY)
    
    cards = [
        ("COMPUTE ENGINE", "240 MHz", "Dual-Core Xtensa LX7", "• Bare-metal C++ execution (Zero framework bloat)\n• SIMD-aligned vector dot products\n• Deterministic timing with zero garbage collection", BLUE_PRIMARY, BLUE_BG, BLUE_BORDER),
        ("MEMORY ALLOCATION", "8 MB", "Octal PSRAM (80MHz)", "• 1.79 MB INT8 Model Weights in PSRAM\n• Dynamic 128-token Key-Value (KV) Cache\n• Zero-wait working buffers for multi-head attention", AMBER_PRIMARY, AMBER_BG, AMBER_BORDER),
        ("MODEL ARCHITECTURE", "1.84M", "Causal Transformer Parameters", "• 4 Transformer Layers | 192 Dim | 4 Heads\n• 91 Vocabulary Tokens\n• 100% Offline execution with zero telemetry", GREEN_PRIMARY, GREEN_BG, GREEN_BORDER),
    ]
    
    y = 220
    for tag, big_stat, subtitle, desc, col, bg_col, b_col in cards:
        draw_shadowed_card(draw, [60, y, 1020, y + 285], radius=16, fill=CARD_BG, outline=CARD_BORDER, width=1)
        draw_badge(draw, 90, y + 22, tag, get_font(15, bold=True), col, bg_col, b_col)
        
        draw.text((90, y + 70), big_stat, font=get_font(34, bold=True, mono=True), fill=col)
        stat_w = get_font(34, bold=True, mono=True).getbbox(big_stat)[2]
        draw.text((105 + stat_w, y + 78), f"—  {subtitle}", font=get_font(22, bold=True), fill=TEXT_PRIMARY)
        
        draw.line([90, y + 118, 990, y + 118], fill=CARD_BORDER, width=1)
        draw.text((90, y + 135), desc, font=get_font(20), fill=TEXT_SECONDARY)
        y += 315
        
    img.save(os.path.join(OUTPUT_DIR, "slide_03.png"))
    print("Slide 3 rendered")

# ==========================================
# SLIDE 4: THE 5-STEP PIPELINE
# ==========================================
def render_slide_4():
    img, draw = create_base_slide("ENGINEERING WORKFLOW", 4)
    
    t_font = get_font(48, bold=True)
    draw.text((60, 130), "From PyTorch to Bare-Metal C++", font=t_font, fill=TEXT_PRIMARY)
    
    steps = [
        ("01", "Dataset Preparation", "Curated custom conversational dataset for companion robot interaction."),
        ("02", "PyTorch Training", "Trained 4-Layer Causal Transformer (1.84M params, 192 dim)."),
        ("03", "INT8 Quantization", "Compressed model from 7.02 MB -> 1.79 MB (100% FP32 fidelity)."),
        ("04", "Bare-Metal C++ Engine", "Custom forward pass and KV-cache without runtime framework bloat."),
        ("05", "Hardware ALU Intercept", "Math questions ('100 x 7') evaluated on hardware ALU in <0.1 ms."),
    ]
    
    y = 220
    for num, title, desc in steps:
        draw_shadowed_card(draw, [60, y, 1020, y + 165], radius=14, fill=CARD_BG, outline=CARD_BORDER, width=1)
        draw.rounded_rectangle([60, y, 70, y + 165], radius=4, fill=BLUE_PRIMARY)
        
        draw.rounded_rectangle([95, y + 22, 165, y + 92], radius=10, fill=BLUE_BG, outline=BLUE_BORDER, width=1)
        draw.text((108, y + 33), num, font=get_font(26, bold=True, mono=True), fill=BLUE_PRIMARY)
        
        draw.text((190, y + 25), title, font=get_font(25, bold=True), fill=TEXT_PRIMARY)
        draw.text((190, y + 72), desc, font=get_font(20), fill=TEXT_SECONDARY)
        y += 185
        
    img.save(os.path.join(OUTPUT_DIR, "slide_04.png"))
    print("Slide 4 rendered")

# ==========================================
# SLIDE 5: CLEAN BENCHMARK TABLE (FP32, FP16, INT8, INT4)
# ==========================================
def render_slide_5():
    img, draw = create_base_slide("QUANTIZATION BENCHMARK", 5)
    
    t_font = get_font(48, bold=True)
    draw.text((60, 130), "Model Quantization Benchmark", font=t_font, fill=TEXT_PRIMARY)
    
    sub_font = get_font(21, bold=False)
    draw.text((60, 195), "Evaluation across 4 precision formats on 1.84M parameters:", font=sub_font, fill=TEXT_SECONDARY)
    
    # Main Table Outer Card
    draw_shadowed_card(draw, [60, 245, 1020, 920], radius=16, fill=CARD_BG, outline=CARD_BORDER_DARK, width=1)
    
    # Table Header Row
    draw.rounded_rectangle([60, 245, 1020, 315], radius=16, fill=(241, 245, 249))
    draw.text((90, 268), "Format", font=get_font(18, bold=True), fill=TEXT_PRIMARY)
    draw.text((250, 268), "Model Size", font=get_font(18, bold=True), fill=TEXT_PRIMARY)
    draw.text((430, 268), "Compression", font=get_font(18, bold=True), fill=TEXT_PRIMARY)
    draw.text((630, 268), "Cosine Sim", font=get_font(18, bold=True), fill=TEXT_PRIMARY)
    draw.text((820, 268), "Status", font=get_font(18, bold=True), fill=TEXT_PRIMARY)
    
    rows = [
        ("FP32", "7.02 MB", "0.0% (Base)", "100.00%", "High PSRAM load", TEXT_SECONDARY, CARD_BG, CARD_BORDER, False),
        ("FP16", "3.51 MB", "50.0%", "100.00%", "Software emulated", TEXT_SECONDARY, CARD_BG, CARD_BORDER, False),
        ("INT8", "1.79 MB", "74.4%", "100.00%", "Selected (Optimal)", BLUE_PRIMARY, BLUE_BG, BLUE_BORDER, True),
        ("INT4", "0.88 MB", "87.2%", "98.87%", "+26% unpack latency", AMBER_PRIMARY, CARD_BG, CARD_BORDER, False),
    ]
    
    ry = 325
    for fmt, size, comp, sim, status, col, bg_col, b_col, is_selected in rows:
        if is_selected:
            draw.rounded_rectangle([70, ry, 1010, ry + 125], radius=12, fill=bg_col, outline=b_col, width=2)
            draw.text((90, ry + 25), fmt, font=get_font(28, bold=True, mono=True), fill=col)
            draw.text((90, ry + 75), "(Selected)", font=get_font(15, bold=True), fill=col)
            draw.text((250, ry + 45), size, font=get_font(24, bold=True), fill=TEXT_PRIMARY)
            draw.text((430, ry + 45), comp, font=get_font(24, bold=True), fill=TEXT_PRIMARY)
            draw.text((630, ry + 45), sim, font=get_font(24, bold=True), fill=GREEN_PRIMARY)
            draw.text((820, ry + 45), status, font=get_font(20, bold=True), fill=col)
        else:
            draw.rounded_rectangle([70, ry, 1010, ry + 125], radius=12, fill=bg_col, outline=b_col, width=1)
            draw.text((90, ry + 45), fmt, font=get_font(26, bold=True, mono=True), fill=col)
            draw.text((250, ry + 45), size, font=get_font(23), fill=TEXT_PRIMARY)
            draw.text((430, ry + 45), comp, font=get_font(23), fill=TEXT_PRIMARY)
            draw.text((630, ry + 45), sim, font=get_font(23), fill=GREEN_PRIMARY if "100" in sim else TEXT_SECONDARY)
            draw.text((820, ry + 45), status, font=get_font(18), fill=TEXT_SECONDARY)
        ry += 140
        
    # Bottom Key Finding Card
    draw_shadowed_card(draw, [60, 945, 1020, 1165], radius=16, fill=CARD_BG, outline=BLUE_BORDER, width=2)
    draw_badge(draw, 90, 970, "KEY BENCHMARK TAKEAWAY", get_font(15, bold=True), BLUE_PRIMARY, BLUE_BG, BLUE_BORDER)
    draw.text((90, 1025), "INT8 Delivers the Optimal Balance for ESP32-S3", font=get_font(24, bold=True), fill=TEXT_PRIMARY)
    takeaway = (
        "• INT8 reduces PSRAM footprint by 74.4% while preserving 100.00% FP32 output quality.\n"
        "• Byte-aligned SIMD execution provides maximum inference throughput (~12-13 tok/s)."
    )
    draw.text((90, 1070), takeaway, font=get_font(19), fill=TEXT_SECONDARY)
    
    img.save(os.path.join(OUTPUT_DIR, "slide_05.png"))
    print("Slide 5 rendered")

# ==========================================
# SLIDE 6: CROPPED TERMINAL
# ==========================================
def render_slide_6():
    img, draw = create_base_slide("LIVE TERMINAL INFERENCE", 6)
    
    t_font = get_font(46, bold=True)
    draw.text((60, 130), "Real-Time UART Generation (~12-13 tok/s)", font=t_font, fill=TEXT_PRIMARY)
    
    draw_shadowed_card(draw, [55, 195, 1025, 865], radius=16, fill=(15, 23, 42), outline=CARD_BORDER_DARK, width=2)
    
    draw.rounded_rectangle([55, 195, 1025, 240], radius=16, fill=(30, 41, 59))
    draw.ellipse([75, 211, 91, 227], fill=(239, 68, 68))
    draw.ellipse([99, 211, 115, 227], fill=(245, 158, 11))
    draw.ellipse([123, 211, 139, 227], fill=(16, 185, 129))
    draw.text((440, 210), "serial_chat.py - 115200 baud", font=get_font(16, bold=True, mono=True), fill=(148, 163, 184))
    
    crop_path = "/home/aiot/.gemini/antigravity-ide/brain/b745120a-0087-4bb2-b22a-7e8180b591c4/scratch/crop_terminal_perfect.jpg"
    if os.path.exists(crop_path):
        term_img = Image.open(crop_path).convert("RGB")
        term_img = term_img.resize((960, 610), Image.Resampling.LANCZOS)
        img.paste(term_img, (60, 245))
        
    metrics = [
        ("12.2 tok/s", "Real-Time Speed", "Physical chip verified", BLUE_PRIMARY, BLUE_BG, BLUE_BORDER),
        ("1.79 MB", "PSRAM Footprint", "8MB Octal PSRAM", AMBER_PRIMARY, AMBER_BG, AMBER_BORDER),
        ("< 0.1 ms", "ALU Math Tool", "Zero hallucinations", GREEN_PRIMARY, GREEN_BG, GREEN_BORDER),
    ]
    
    x = 60
    for val, label, sub, col, bg_col, b_col in metrics:
        draw_shadowed_card(draw, [x, 890, x + 300, 1165], radius=14, fill=CARD_BG, outline=b_col, width=1)
        draw.text((x + 25, 920), val, font=get_font(34, bold=True, mono=True), fill=col)
        draw.text((x + 25, 980), label, font=get_font(22, bold=True), fill=TEXT_PRIMARY)
        draw.text((x + 25, 1025), sub, font=get_font(18), fill=TEXT_MUTED)
        x += 330
        
    img.save(os.path.join(OUTPUT_DIR, "slide_06.png"))
    print("Slide 6 rendered")

# ==========================================
# SLIDE 7: MULTI-BOARD & OPEN SOURCE
# ==========================================
def render_slide_7():
    img, draw = create_base_slide("MULTI-BOARD & OPEN SOURCE", 7)
    
    t_font = get_font(48, bold=True)
    draw.text((60, 130), "Validated on 3 Form Factors", font=t_font, fill=TEXT_PRIMARY)
    
    boards = [
        ("ESP32-S3 DevKit N16R8", "16MB Flash | 8MB Octal PSRAM", "esp32s3_devkit_tight_horiz.jpg"),
        ("Arduino Nano ESP32", "16MB Flash | 8MB Octal PSRAM", "arduino_nano_esp32.jpg"),
        ("Seeed Studio XIAO S3", "8MB Flash | 8MB Octal PSRAM", "seeed_xiao_esp32s3.jpg"),
    ]
    
    y = 215
    for name, specs, photo_file in boards:
        draw_shadowed_card(draw, [60, y, 1020, y + 195], radius=16, fill=CARD_BG, outline=CARD_BORDER, width=1)
        
        photo_path = os.path.join(ASSETS_DIR, photo_file)
        if os.path.exists(photo_path):
            b_img = Image.open(photo_path).convert("RGB")
            b_img = b_img.resize((210, 92) if "devkit" in photo_file else (210, 155), Image.Resampling.LANCZOS)
            img.paste(b_img, (80, y + 50 if "devkit" in photo_file else y + 20))
            
        draw.text((320, y + 35), name, font=get_font(27, bold=True), fill=TEXT_PRIMARY)
        draw.text((320, y + 80), specs, font=get_font(21), fill=BLUE_PRIMARY)
        draw.text((320, y + 125), "✓ 100% Tested & Verified on Hardware", font=get_font(18, bold=True), fill=GREEN_PRIMARY)
        y += 225
        
    draw_shadowed_card(draw, [60, 915, 1020, 1165], radius=16, fill=BLUE_BG, outline=BLUE_BORDER, width=2)
    draw.text((95, 940), "100% Open Source on GitHub (MIT License)", font=get_font(25, bold=True), fill=BLUE_PRIMARY)
    draw.text((95, 985), "github.com/fitranurmayadi/esp32-microlm", font=get_font(25, bold=True, mono=True), fill=TEXT_PRIMARY)
    draw.text((95, 1045), "Clone, train your custom dataset, and flash to your ESP32!", font=get_font(20), fill=TEXT_SECONDARY)
    
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
print(f"All 7 Minimalist Light Mode slides and PDF updated at {pdf_path}")
