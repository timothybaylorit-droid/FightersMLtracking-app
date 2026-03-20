import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

st.title("🥊 AI Fight Tracker with Trails + Heatmap")

# 1️⃣ Upload video
uploaded_file = st.file_uploader("Upload a fight video (MP4)", type=["mp4"])

if uploaded_file:
    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    st.success("File uploaded! Processing video...")

    # Load YOLO model
    model = YOLO("yolov8n.pt")

    # Open video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output video
    output_file = "processed_fight.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Initialize tracking paths
    fighter_paths = {"A": [], "B": []}
    stframe = st.empty()
    frame_count = 0
    display_every_n_frames = 5  # show every 5 frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for detection
        frame_resized = cv2.resize(frame, (640, 360))

        # Detect people
        results = model(frame_resized)
        people = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    people.append((cx, cy, x1, y1, x2, y2))

        people = sorted(people, key=lambda x: x[0])

        if len(people) >= 2:
            fighter_A = people[0]
            fighter_B = people[1]

            fighter_paths["A"].append((fighter_A[0], fighter_A[1]))
            fighter_paths["B"].append((fighter_B[0], fighter_B[1]))

            # Scale back to original resolution
            scale_x = width / 640
            scale_y = height / 360
            cv2.rectangle(frame,
                          (int(fighter_A[2]*scale_x), int(fighter_A[3]*scale_y)),
                          (int(fighter_A[4]*scale_x), int(fighter_A[5]*scale_y)),
                          (255,0,0), 2)
            cv2.rectangle(frame,
                          (int(fighter_B[2]*scale_x), int(fighter_B[3]*scale_y)),
                          (int(fighter_B[4]*scale_x), int(fighter_B[5]*scale_y)),
                          (0,0,255), 2)

        # Draw trails
        for i in range(1, len(fighter_paths["A"])):
            pt1 = (int(fighter_paths["A"][i-1][0]*scale_x), int(fighter_paths["A"][i-1][1]*scale_y))
            pt2 = (int(fighter_paths["A"][i][0]*scale_x), int(fighter_paths["A"][i][1]*scale_y))
            cv2.line(frame, pt1, pt2, (255,0,0), 2)
        for i in range(1, len(fighter_paths["B"])):
            pt1 = (int(fighter_paths["B"][i-1][0]*scale_x), int(fighter_paths["B"][i-1][1]*scale_y))
            pt2 = (int(fighter_paths["B"][i][0]*scale_x), int(fighter_paths["B"][i][1]*scale_y))
            cv2.line(frame, pt1, pt2, (0,0,255), 2)

        # Optional: overlay heatmap
        if len(fighter_paths["A"]) > 10:
            heatmap = np.zeros((360,640), dtype=np.float32)
            for pt in fighter_paths["A"] + fighter_paths["B"]:
                x = int(pt[0] * 640 / 640)
                y = int(pt[1] * 360 / 360)
                heatmap[y, x] += 1
            heatmap = np.clip(heatmap*10,0,255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            frame = cv2.addWeighted(frame, 0.8, heatmap_color, 0.2, 0)

        # Write full-res video
        out.write(frame)

        # Show every N frames
        if frame_count % display_every_n_frames == 0:
            frame_display = cv2.resize(frame, (640,360))
            stframe.image(frame_display, channels="BGR")

        frame_count += 1

    cap.release()
    out.release()
    st.success("Processing finished!")

    # Download button
    with open(output_file, "rb") as f:
        st.download_button("Download Processed Video", f, file_name="processed_fight.mp4")