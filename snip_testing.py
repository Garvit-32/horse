import os
import glob
import cv2
import torch
import numpy as np
import copy

mp4s = glob.glob("./sanchit_data/*.mp4")
pts = [k.replace(".mp4", "_results.pt") for k in mp4s]


def get_horse_orientation(kp3d, conf=0.5) -> str:
    # Use a few keypoints indices to decide orientation.
    front_indices = [0, 1, 2]  # Example: Head area
    back_indices = [4, 14, 11] # Example: Tail/Rear area

    if isinstance(kp3d, torch.Tensor):
        kp3d = kp3d.cpu().numpy()

    # Ensure indices are valid
    valid_front_indices = [i for i in front_indices if i < kp3d.shape[0]]
    valid_back_indices = [i for i in back_indices if i < kp3d.shape[0]]

    if not valid_front_indices or not valid_back_indices:
        return "unknown" # Not enough keypoints

    front_coords = kp3d[valid_front_indices]
    back_coords = kp3d[valid_back_indices]

    # Optional: Add confidence check here if kp3d includes confidence scores
    # e.g., filter coords based on a threshold

    front_mean = np.mean(front_coords, axis=0)
    back_mean = np.mean(back_coords, axis=0)

    diff = front_mean - back_mean
    # Check if diff is valid (might be NaN if coords are bad)
    if np.isnan(diff).any():
        return "unknown"

    dx, _, dz = diff # Use x (left/right) and z (front/back relative to camera)

    # Handle cases where dz is very small to avoid instability
    if abs(dz) < 1e-6:
        dz = 1e-6

    # If horizontal difference dominates, use left/right decision
    if abs(dx) > conf * abs(dz):
        return "left" if dx < 0 else "right" # dx < 0 means front is to the left
    else:
        # dz < 0 means front is closer to the camera ("front" view)
        # dz > 0 means front is further from the camera ("back" view)
        return "front" if dz < 0 else "back"

def snip_consecutive(arr, min_len=15, smooth_threshold=3):
    """
    Divides an array into subarrays based on consecutive values,
    applying smoothing and minimum length rules.

    Args:
        arr (list): The input list of values (e.g., strings).
        min_len (int): The minimum length required for a final segment.
        smooth_threshold (int): The maximum length of a segment to be
                                 "smoothed over" if surrounded by identical values.

    Returns:
        dict: A dictionary where keys are (start_index, end_index) tuples
              and values are the corresponding segment value.
    """

    if not arr:
        return {}

    # --- Step 1: Identify all initial consecutive segments ---
    initial_segments = []
    if arr: # Ensure array is not empty
        current_val = arr[0]
        current_start = 0
        for i in range(1, len(arr)):
            if arr[i] != current_val:
                initial_segments.append([current_val, current_start, i - 1]) # Use list for mutability
                current_val = arr[i]
                current_start = i
        # Add the last segment
        initial_segments.append([current_val, current_start, len(arr) - 1])

    if not initial_segments:
        return {}

    smoothed_segments = copy.deepcopy(initial_segments) # Work on a copy

    # Iterate considering segments i-1, i, i+1
    for i in range(1, len(smoothed_segments) - 1):
        prev_val, _, _ = smoothed_segments[i-1]
        curr_val, curr_start, curr_end = smoothed_segments[i]
        next_val, _, _ = smoothed_segments[i+1]

        length = curr_end - curr_start + 1

        # Check if current segment is short and surrounded by same value
        if length <= smooth_threshold and prev_val == next_val:
            # Change the value of the middle segment to match neighbours
            smoothed_segments[i][0] = prev_val

    # --- Step 3: Merge Adjacent Segments with the Same Value ---
    if not smoothed_segments:
        return {}

    merged_segments = []
    # Start with the first segment from the smoothed list
    current_merge_val, current_merge_start, current_merge_end = smoothed_segments[0]

    for i in range(1, len(smoothed_segments)):
        next_val, next_start, next_end = smoothed_segments[i]

        if next_val == current_merge_val:
            # If the next segment has the same value, extend the current merged segment
            current_merge_end = next_end
        else:
            # If the value changes, store the completed merged segment
            merged_segments.append((current_merge_val, current_merge_start, current_merge_end))
            # Start a new merged segment
            current_merge_val = next_val
            current_merge_start = next_start
            current_merge_end = next_end

    # Add the last merged segment
    merged_segments.append((current_merge_val, current_merge_start, current_merge_end))

    # --- Step 4: Filter by Minimum Length and Format Output ---
    result = {}
    for val, start_idx, end_idx in merged_segments:
        length = end_idx - start_idx + 1
        if length >= min_len:
            result[(start_idx, end_idx)] = val

    return result

def frame_extract(vid):
    frames = []
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        yield frame

orientation_kp = "Unknown"

from tqdm import tqdm

os.makedirs("snipped_videos", exist_ok=True)

for mp4, pt_file in tqdm(zip(mp4s, pts), total=len(mp4s)):
    vid = cv2.VideoCapture(mp4)
    if not vid.isOpened():
        print(f"Error opening video file {mp4}")
        continue
    pt = torch.load(pt_file, map_location=torch.device('cpu'))
    kp_3d = pt["pred_kp3d_crop"]

    output = []
    for idx, frame in enumerate(frame_extract(vid)):
        if idx % 3 == 0:
            tensor_idx = idx // 3
            if tensor_idx >= len(kp_3d):
                continue
            kp3d_frame = kp_3d[tensor_idx]
            orientation_kp = get_horse_orientation(kp3d_frame)
            output.append(orientation_kp)

    snipped_output = snip_consecutive(output)

    os.makedirs(os.path.join("snipped_videos", mp4.split("/")[-1].replace(".mp4", "")), exist_ok=True)

    for idx, (frame_range, val) in enumerate(snipped_output.items()):
        start_frame, end_frame = frame_range
        start_frame = start_frame * 3
        end_frame = end_frame * 3 + 3

        # Go to the start frame
        vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        segment_writer = cv2.VideoWriter(
            os.path.join("snipped_videos", mp4.split("/")[-1].replace(".mp4", ""),
                        f"segment_{idx}_{val}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (int(vid.get(3)), int(vid.get(4))),
        )

        # Write frames up to (and including) end_frame
        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = vid.read()
            if not ret:
                break
            segment_writer.write(frame)
        segment_writer.release()
    vid.release()
