"""
Requirements  (Python ≥3.9):
pip install ultralytics opencv-python numpy matplotlib
"""
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm 

def video_frame_generator(cap):
    """Yield frames from an open cv2.VideoCapture until the video ends."""
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame

# ------------------------------------------------------------------
# 1. Load the YOLOv8-segmentation model (pre-trained on COCO-Seg)
# ------------------------------------------------------------------
#  N.B. If you’ve trained your own model, replace "yolov8x-seg.pt"
model = YOLO("yolo11x-seg.pt")      # or "yolov8n-seg.pt" for the nano model

# COCO class IDs
PERSON_CLASS = 0       # “person”
HORSE_CLASS  = 17      # “horse”

# ------------------------------------------------------------------
# 2. Prepare video reader & writer (optional)
# ------------------------------------------------------------------
video_path   = "/workspace/Dessie/dataset/4DCGI_Sale_AUG 20/Lot 11 - Really Chic (USA) B.M. BACK TO LIST/video_1_1_trim.68B92E3F-9123-45A9-8E66-45AA0FD7CC98.mp4"          # ← your video here
save_vis     = True                 # flip to False if you don’t need a vis video
out_path     = Path("iou_visualised.mp4")

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if save_vis:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

# ------------------------------------------------------------------
# 3. Process every frame
# ------------------------------------------------------------------
ious = []
frame_idx = 0

for frame in tqdm(video_frame_generator(cap)):

    # 3a. Run YOLO11‑Seg inference
    r = model(frame, verbose=False)[0]

    # 3b. Fetch masks & class IDs
    masks      = r.masks.data.cpu().numpy()      # shape [N, Hm, Wm]  (~384×640)
    class_ids  = r.boxes.cls.cpu().numpy().astype(int)
    # breakpoint()

    person_mask = horse_mask = None
    for m, cid in zip(masks, class_ids):
        if cid == PERSON_CLASS and person_mask is None:
            person_mask = m >= 0.5                # binarise once
        elif cid == HORSE_CLASS and horse_mask is None:
            horse_mask  = m >= 0.5

    if horse_mask is None :
        print(f"[Frame {frame_idx}] Warning: No horse detected!")
        cv2.imwrite("/workspace/temp.png",frame)

    iou = np.nan

    if person_mask is None:
        iou = 0

    if person_mask is not None and horse_mask is not None:                       # horse mask is always there
        inter = np.logical_and(person_mask, horse_mask).sum()
        union = np.logical_or (person_mask, horse_mask).sum()
        iou   = inter / union if union else np.nan
    ious.append(iou)
    # breakpoint()

    # 3d. OPTIONAL visualisation ---------------------------------------
    if save_vis and horse_mask is not None:
        # print(horse_mask.shape)
        mask_h, mask_w = horse_mask.shape

        # down‑scale the RGB frame once to mask size
        frame_small = cv2.resize(frame, (mask_w, mask_h), interpolation=cv2.INTER_AREA)
        vis = frame_small.copy()
        alpha = 0.45

        # draw horse (if detected)
        vis[horse_mask] = ((1 - alpha) * vis[horse_mask] + alpha * np.array((255, 0, 0))).astype(np.uint8)

        # draw person (if detected)
        if person_mask is not None:
            vis[person_mask] = ((1 - alpha) * vis[person_mask] + alpha * np.array((0, 255, 0))).astype(np.uint8)

        cv2.putText(vis, f"F{frame_idx:05d}  IoU:{iou:.3f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # upscale back so the output video keeps the original resolution
        writer.write(cv2.resize(vis, (w, h)))

    elif save_vis:
        # fallback when horse is missing: show raw frame
        print(f"[Frame {frame_idx:05d}] Warning: No horse detected for visualisation")
        writer.write(frame)

    frame_idx += 1

cap.release()
if save_vis:
    writer.release()

# ------------------------------------------------------------------
# 4. Quick plot of IoU over time  (optional)
# ------------------------------------------------------------------
valid = ~np.isnan(ious)
if valid.any():
    plt.figure()
    plt.plot(np.where(valid)[0], np.array(ious)[valid], linewidth=2)
    plt.title("Person–Horse IoU per frame")
    plt.xlabel("Frame index")
    plt.ylabel("IoU")
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig("iou_plot.png")  # or use .pdf / .jpg etc.
    print("Saved IoU plot to: iou_plot.png")

    plt.show()
else:
    print("No frames contained BOTH a person and a horse – nothing to plot.")

# 5. Save raw IoU values if you need them elsewhere
np.save("frame_iou_values.npy", np.array(ious))
print("Done!   IoU values saved to frame_iou_values.npy")
if save_vis:
    print(f"Visualised video saved to: {out_path.resolve()}")
