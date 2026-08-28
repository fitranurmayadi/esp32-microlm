import os
from PIL import Image, ImageDraw, ImageFont

OUTPUT_PATH = "/home/aiot/Projects/KIBO-MICROLM/testing_video/single_hero_slide.png"
ASSETS_DIR = "/home/aiot/Projects/KIBO-MICROLM/testing_video/assets"

W, H = 1080, 1350

# Clean, Accessible Editorial Palette
BG_CANVAS = (248, 250, 252)     # #F8FAFC
CARD_BG = (255, 255, 255)       # #FFFFFF
CARD_BORDER = (226, 232, 240)   # #E2E8F0
CARD_SHADOW = (238, 242, 246)   # #EEF2F6

TEXT_PRIMARY = (15, 23, 42)     # #0F172A (Deep Slate Black)
TEXT_SECONDARY = (51, 65, 85)   # #334155
TEXT_MUTED = (100, 116, 139)    # #64748B

# Signature Clean Accent: Deep Azure Blue
ACCENT_PRIMARY = (2, 132, 199)  # #0284C7
ACCENT_BG = (240, 249, 255)     # #F0F9FF
ACCENT_BORDER = (186, 230, 253) # #BAE6FD

# Verification Accent: Refined Teal
TEAL_PRIMARY = (13, 148, 136)   # #0D9488
TEAL_BG = (240, 253, 250)       # #F0FDFA
TEAL_BORDER = (153, 246, 228)   # #99F6E4

FONT_REGULAR = "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf"
FONT_MONO = "/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf"

def get_font(size, bold=False, mono=False):
    try:
        if mono and os.path.exists(FONT_MONO):
            return ImageFont.truetype(FONT_MONO, size)
        if bold and os.path.exists(FONT_BOLD):
            return ImageFont.truetype(FONT_BOLD, size)
        if os.path.exists(FONT_REGULAR):
            return ImageFont.truetype(FONT_REGULAR, size)
        return ImageFont.load_default()
    except:
        return ImageFont.load_default()

def draw_shadowed_card(draw, bbox, radius=14, fill=CARD_BG, outline=CARD_BORDER, width=1):
    x0, y0, x1, y1 = bbox
    draw.rounded_rectangle([x0, y0 + 2, x1, y1 + 2], radius=radius, fill=CARD_SHADOW)
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill, outline=outline, width=int(width))

def draw_badge(draw, x, y, text, font, text_color=ACCENT_PRIMARY, bg_color=ACCENT_BG, border_color=ACCENT_BORDER):
    bbox = font.getbbox(text)
    w = bbox[2] - bbox[0] + 20
    h = bbox[3] - bbox[1] + 14
    draw.rounded_rectangle([x, y, x + w, y + h], radius=6, fill=bg_color, outline=border_color, width=1)
    draw.text((x + 10, y + 6), text, font=font, fill=text_color)
    return w, h

def render_accessible_hero():
    img = Image.new("RGB", (W, H), BG_CANVAS)
    draw = ImageDraw.Draw(img)
    
    # 1. Top Badges (Simple & Clear)
    draw_badge(draw, 60, 48, "ON-DEVICE AI", get_font(14, bold=True), ACCENT_PRIMARY, ACCENT_BG, ACCENT_BORDER)
    draw_badge(draw, W - 220, 48, "OPEN SOURCE", get_font(14, bold=True), TEAL_PRIMARY, TEAL_BG, TEAL_BORDER)
    
    # 2. Main Title & Clear Subtitle
    draw.text((60, 110), "Running a Language Model", font=get_font(44, bold=True), fill=TEXT_PRIMARY)
    draw.text((60, 170), "Entirely on the ESP32-S3", font=get_font(44, bold=True), fill=ACCENT_PRIMARY)
    
    sub = "A self-contained AI brain running locally on a tiny microcontroller"
    draw.text((60, 238), sub, font=get_font(21), fill=TEXT_SECONDARY)
    
    # 3. Main Center Grid
    # Left Card: Hardware Photo with Simple Specs
    draw_shadowed_card(draw, [60, 295, 475, 930], radius=16, fill=CARD_BG, outline=CARD_BORDER, width=1)
    
    dev_path = os.path.join(ASSETS_DIR, "esp32s3_devkit_tight.jpg")
    if os.path.exists(dev_path):
        hw_img = Image.open(dev_path).convert("RGB")
        hw_img = hw_img.resize((225, 520), Image.Resampling.LANCZOS)
        img.paste(hw_img, (80, 340))
        
    draw_badge(draw, 325, 340, "MICROCONTROLLER", get_font(12, bold=True), ACCENT_PRIMARY, ACCENT_BG, ACCENT_BORDER)
    draw.text((325, 385), "ESP32-S3", font=get_font(23, bold=True), fill=TEXT_PRIMARY)
    draw.text((325, 420), "DevKit N16R8\n\n• 240 MHz Speed\n• 8MB RAM\n• 16MB Storage\n• 100% Offline", font=get_font(16), fill=TEXT_SECONDARY)
    
    # Right Side: 4 Clear, Human-Understandable Feature Cards (Zero Jargon / Zero Repetition)
    pillars = [
        ("MODEL CAPACITY", "1.84 Million Parameters", "Custom 4-layer neural network designed for on-device chat."),
        ("MEMORY FOOTPRINT", "1.79 MB Compressed Size", "Shrunk by 74% to fit easily inside the chip's internal RAM."),
        ("RESPONSE SPEED", "~12 Tokens / Second", "Fast real-time streaming text without needing the internet."),
        ("SMART CALCULATOR", "Instant Math Engine", "Solves arithmetic questions in <0.1 ms with 100% accuracy."),
    ]
    
    ry = 295
    for tag, title, desc in pillars:
        draw_shadowed_card(draw, [495, ry, 1020, ry + 145], radius=14, fill=CARD_BG, outline=CARD_BORDER, width=1)
        draw_badge(draw, 520, ry + 16, tag, get_font(12, bold=True), ACCENT_PRIMARY, ACCENT_BG, ACCENT_BORDER)
        draw.text((520, ry + 52), title, font=get_font(25, bold=True), fill=TEXT_PRIMARY)
        draw.text((520, ry + 94), desc, font=get_font(17), fill=TEXT_SECONDARY)
        ry += 158
        
    # 4. Bottom Section: Simple 3-Step Creation Pipeline
    draw_shadowed_card(draw, [60, 955, 1020, 1225], radius=16, fill=CARD_BG, outline=CARD_BORDER, width=1)
    draw_badge(draw, 85, 975, "HOW IT WAS BUILT", get_font(13, bold=True), ACCENT_PRIMARY, ACCENT_BG, ACCENT_BORDER)
    draw.text((275, 982), "From Model Training to Hardware Deployment", font=get_font(17, bold=True), fill=TEXT_PRIMARY)
    
    steps = [
        ("1. Train Model", "train.py", "Teach the model custom\ndialogues and vocabulary."),
        ("2. Compress Weights", "export_int8_bin.py", "Pack the parameters into\na compact binary file."),
        ("3. Flash to Board", "kibo_esp32.ino", "Upload standalone C++ code\nand chat live over USB."),
    ]
    
    bx = 85
    for step_title, step_code, step_desc in steps:
        draw.rounded_rectangle([bx, 1030, bx + 285, 1195], radius=12, fill=(248, 250, 252), outline=CARD_BORDER, width=1)
        draw.text((bx + 16, 1048), step_title, font=get_font(17, bold=True), fill=TEXT_PRIMARY)
        draw.text((bx + 16, 1082), step_code, font=get_font(15, bold=True, mono=True), fill=ACCENT_PRIMARY)
        draw.text((bx + 16, 1120), step_desc, font=get_font(13), fill=TEXT_SECONDARY)
        bx += 310
        
    # 5. Clean Distinct Footer
    draw.line([60, 1260, 1020, 1260], fill=CARD_BORDER, width=1)
    draw.text((60, 1285), "Fitra Nurmayadi  |  ESP32 Micro-LM", font=get_font(19, bold=True), fill=TEXT_PRIMARY)
    
    cta_badge = "github.com/fitranurmayadi/esp32-microlm"
    draw.text((W - 480, 1285), cta_badge, font=get_font(20, bold=True, mono=True), fill=ACCENT_PRIMARY)
    
    img.save(OUTPUT_PATH)
    print(f"Accessible & Non-Technical Single Hero Slide created at {OUTPUT_PATH}")

if __name__ == "__main__":
    render_accessible_hero()
