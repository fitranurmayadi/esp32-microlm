import os
import cv2
import numpy as np
from PIL import Image

TEMPLATE_PATH = "/home/aiot/Projects/KIBO-MICROLM/testing_video/assets/slide_06_template.png"
VIDEO_PATH = "/home/aiot/Projects/KIBO-MICROLM/testing_video/kibo_terminal_cropped.mp4"
OUTPUT_MP4 = "/home/aiot/Projects/KIBO-MICROLM/testing_video/carousel_slides/slide_06_animated.mp4"
OUTPUT_GIF = "/home/aiot/Projects/KIBO-MICROLM/testing_video/carousel_slides/slide_06_animated.gif"

base_img = Image.open(TEMPLATE_PATH).convert("RGB")
base_np = np.array(base_img)
H, W, _ = base_np.shape

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Reading video: {total_frames} frames @ {fps} fps")

# Prepare video writer for MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_MP4, fourcc, fps, (W, H))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Frame is BGR from OpenCV, resize to (960, 610)
    frame_resized = cv2.resize(frame, (960, 610), interpolation=cv2.INTER_LANCZOS4)
    
    # Composite into base slide
    composite = cv2.cvtColor(base_np, cv2.COLOR_RGB2BGR).copy()
    composite[245:245+610, 60:60+960] = frame_resized
    
    out.write(composite)
    frame_count += 1
    if frame_count % 300 == 0:
        print(f"Rendered {frame_count}/{total_frames} frames...")

cap.release()
out.release()
print(f"MP4 rendered to {OUTPUT_MP4}")

# Now convert to optimized H.264 mp4 and animated GIF using ffmpeg
os.system(f"ffmpeg -y -i '{OUTPUT_MP4}' -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p '{OUTPUT_MP4}.temp.mp4' && mv '{OUTPUT_MP4}.temp.mp4' '{OUTPUT_MP4}'")

# Generate optimized GIF (scaled to 720px width for fast web playback)
os.system(f"ffmpeg -y -i '{OUTPUT_MP4}' -vf 'fps=10,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse' -loop 0 '{OUTPUT_GIF}'")

print(f"GIF rendered to {OUTPUT_GIF}")
