import streamlit as st
import mediapipe as mp
import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip
from PIL import Image, ImageDraw
import tempfile

st.title("🥊 Mobile-Friendly AI Fight Tracker (Mediapipe)")

# --- Upload video ---
uploaded_file = st.file_uploader("Upload a fight video (MP4)", type=["mp4"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    st.success("Video uploaded! Processing...")

    # Mediapipe pose setup
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    # Load video with MoviePy
    clip = VideoFileClip(video_path)
    frames = list(clip.iter_frames(fps=10))  # downsample for speed
    st.info(f"Video has {len(frames)} frames at 10 fps for processing.")

    fighter_paths = {"A": [], "B": []}
    processed_frames = []

    for idx, frame in enumerate(frames):
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)

        # --- Pose detection ---
        frame_rgb = np.array(img_pil)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Take left and right hip as a proxy for each fighter's center
            lm = results.pose_landmarks.landmark
            height, width, _ = frame_rgb.shape
            left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

            # Convert normalized coordinates to pixels
            fighter_A_pos = (int(left_hip.x * width), int(left_hip.y * height))
            fighter_B_pos = (int(right_hip.x * width), int(right_hip.y * height))

            fighter_paths["A"].append(fighter_A_pos)
            fighter_paths["B"].append(fighter_B_pos)

            # Draw simple circles for fighter positions
            draw.ellipse([fighter_A_pos[0]-10, fighter_A_pos[1]-10,
                          fighter_A_pos[0]+10, fighter_A_pos[1]+10],
                         fill="blue")
            draw.ellipse([fighter_B_pos[0]-10, fighter_B_pos[1]-10,
                          fighter_B_pos[0]+10, fighter_B_pos[1]+10],
                         fill="red")

        # Draw trails
        for path, color in zip(["A", "B"], ["blue", "red"]):
            points = fighter_paths[path]
            if len(points) > 1:
                draw.line(points, fill=color, width=2)

        # Heatmap overlay
        if len(fighter_paths["A"]) > 5:
            heatmap = np.zeros((img_pil.height, img_pil.width), dtype=np.float32)
            for pt in fighter_paths["A"] + fighter_paths["B"]:
                x = min(pt[0], img_pil.width-1)
                y = min(pt[1], img_pil.height-1)
                heatmap[y, x] += 1
            heatmap = np.clip(heatmap * 10, 0, 255).astype(np.uint8)
            heatmap_img = Image.fromarray(heatmap).convert("RGB")
            heatmap_img = heatmap_img.resize(img_pil.size)
            img_pil = Image.blend(img_pil, heatmap_img, alpha=0.2)

        processed_frames.append(np.array(img_pil))

        if idx % 10 == 0:
            st.image(np.array(img_pil), caption=f"Frame {idx+1}/{len(frames)}")

    # Write processed video
    output_file = "processed_fight_mediapipe.mp4"
    clip_out = ImageSequenceClip(processed_frames, fps=10)
    clip_out.write_videofile(output_file, codec="libx264")

    # Download button
    with open(output_file, "rb") as f:
        st.download_button("Download Processed Video", f, file_name="processed_fight_mediapipe.mp4")

    st.success("Processing complete! ✅")