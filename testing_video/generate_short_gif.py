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

# Exact segment: 16.5s to 25.5s (9.0 seconds)
start_frame = int(16.5 * fps)
end_frame = int(25.5 * fps)

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
total_frames = end_frame - start_frame

print(f"Rendering 9.0s segment ('how are you?'): {total_frames} frames @ {fps} fps...")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_MP4, fourcc, fps, (W, H))

for _ in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_resized = cv2.resize(frame, (960, 610), interpolation=cv2.INTER_LANCZOS4)
    composite = cv2.cvtColor(base_np, cv2.COLOR_RGB2BGR).copy()
    composite[245:245+610, 60:60+960] = frame_resized
    
    out.write(composite)

cap.release()
out.release()
print(f"Short MP4 rendered: {total_frames} frames")

# 1. Optimize MP4
os.system(f"ffmpeg -y -i '{OUTPUT_MP4}' -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p '{OUTPUT_MP4}.temp.mp4' && mv '{OUTPUT_MP4}.temp.mp4' '{OUTPUT_MP4}'")

# 2. Generate Slide 6 GIF (9s @ 12 fps, scaled 720w)
os.system(f"ffmpeg -y -i '{OUTPUT_MP4}' -vf 'fps=12,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse' -loop 0 '{OUTPUT_GIF}'")

# 3. Generate standalone cropped terminal GIF for README
os.system(f"ffmpeg -y -ss 16.5 -t 9.0 -i '{VIDEO_PATH}' -vf 'fps=12,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse' -loop 0 '{SHORT_TERM_GIF}'")

print("9.0s 'how are you?' GIF & MP4 assets successfully created!")
