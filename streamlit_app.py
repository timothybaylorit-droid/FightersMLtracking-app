import streamlit as st
import numpy as np
from ultralytics import YOLO
from moviepy.editor import VideoFileClip, ImageSequenceClip
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import tempfile

st.title("🥊 Light AI Fight Tracker (No OpenCV)")

# --- Upload video ---
uploaded_file = st.file_uploader("Upload a fight video (MP4)", type=["mp4"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    st.success("Video uploaded! Processing...")

    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")

    # --- Load video using MoviePy ---
    clip = VideoFileClip(video_path)
    frames = list(clip.iter_frames(fps=10))  # downsample for speed
    st.info(f"Video has {len(frames)} frames at 10 fps for processing.")

    fighter_paths = {"A": [], "B": []}
    processed_frames = []

    # --- Process frames ---
    for idx, frame in enumerate(frames):
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)

        # YOLO detection
        results = model(frame)
        people = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # class 0 = person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    people.append((cx, cy, x1, y1, x2, y2))

        # Sort by x position to identify Fighter A / B
        people = sorted(people, key=lambda x: x[0])

        if len(people) >= 2:
            fighter_A = people[0]
            fighter_B = people[1]

            fighter_paths["A"].append((fighter_A[0], fighter_A[1]))
            fighter_paths["B"].append((fighter_B[0], fighter_B[1]))

            # Draw bounding boxes
            draw.rectangle([fighter_A[2], fighter_A[3], fighter_A[4], fighter_A[5]], outline="blue", width=3)
            draw.rectangle([fighter_B[2], fighter_B[3], fighter_B[4], fighter_B[5]], outline="red", width=3)

        # Draw trails
        for path, color in zip(["A", "B"], ["blue", "red"]):
            points = fighter_paths[path]
            if len(points) > 1:
                draw.line(points, fill=color, width=2)

        # --- Heatmap overlay ---
        if len(fighter_paths["A"]) > 5:
            heatmap = np.zeros((img_pil.height, img_pil.width), dtype=np.float32)
            for pt in fighter_paths["A"] + fighter_paths["B"]:
                x = min(pt[0], img_pil.width - 1)
                y = min(pt[1], img_pil.height - 1)
                heatmap[y, x] += 1
            heatmap = np.clip(heatmap * 10, 0, 255).astype(np.uint8)
            heatmap_img = Image.fromarray(heatmap).convert("RGB")
            heatmap_img = heatmap_img.resize(img_pil.size)
            img_pil = Image.blend(img_pil, heatmap_img, alpha=0.2)

        processed_frames.append(np.array(img_pil))

        # Display every 10 frames to save mobile bandwidth
        if idx % 10 == 0:
            st.image(np.array(img_pil), caption=f"Frame {idx+1}/{len(frames)}")

    # --- Write processed video ---
    output_file = "processed_fight_light.mp4"
    clip_out = ImageSequenceClip(processed_frames, fps=10)
    clip_out.write_videofile(output_file, codec="libx264")

    # --- Download button ---
    with open(output_file, "rb") as f:
        st.download_button("Download Processed Video", f, file_name="processed_fight_light.mp4")

    st.success("Processing complete! ✅")