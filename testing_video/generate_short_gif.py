import os
import cv2
import numpy as np
from PIL import Image

TEMPLATE_PATH = "/home/aiot/Projects/KIBO-MICROLM/testing_video/assets/slide_06_template.png"
VIDEO_PATH = "/home/aiot/Projects/KIBO-MICROLM/testing_video/kibo_terminal_cropped.mp4"
OUTPUT_MP4 = "/home/aiot/Projects/KIBO-MICROLM/testing_video/carousel_slides/slide_06_animated.mp4"
OUTPUT_GIF = "/home/aiot/Projects/KIBO-MICROLM/testing_video/carousel_slides/slide_06_animated.gif"
SHORT_TERM_GIF = "/home/aiot/Projects/KIBO-MICROLM/testing_video/kibo_terminal_cropped.gif"

base_img = Image.open(TEMPLATE_PATH).convert("RGB")
base_np = np.array(base_img)
H, W, _ = base_np.shape

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

# Limit to first 12.5 seconds (boot + prompt + live generation + stats)
max_frames = int(12.5 * fps)

print(f"Rendering short video: {max_frames} frames (12.5s @ {fps} fps)...")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_MP4, fourcc, fps, (W, H))

frame_count = 0
while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_resized = cv2.resize(frame, (960, 610), interpolation=cv2.INTER_LANCZOS4)
    composite = cv2.cvtColor(base_np, cv2.COLOR_RGB2BGR).copy()
    composite[245:245+610, 60:60+960] = frame_resized
    
    out.write(composite)
    frame_count += 1

cap.release()
out.release()
print(f"Short MP4 rendered: {frame_count} frames")

# 1. Optimize MP4
os.system(f"ffmpeg -y -i '{OUTPUT_MP4}' -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p '{OUTPUT_MP4}.temp.mp4' && mv '{OUTPUT_MP4}.temp.mp4' '{OUTPUT_MP4}'")

# 2. Generate punchy short Slide 6 GIF (12.5s @ 12 fps, scaled 720w)
os.system(f"ffmpeg -y -i '{OUTPUT_MP4}' -vf 'fps=12,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse' -loop 0 '{OUTPUT_GIF}'")

# 3. Generate standalone cropped terminal short GIF for README/Chat
os.system(f"ffmpeg -y -ss 0 -t 12.5 -i '{VIDEO_PATH}' -vf 'fps=12,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse' -loop 0 '{SHORT_TERM_GIF}'")

print("All short GIF and MP4 assets created!")
