import cv2
from ultralytics import YOLO
import numpy as np
import os
from tqdm import tqdm
import multiprocessing as mp
import torch
import json
import time # Optional: for timing diagnostics
import traceback # Import traceback for detailed error logging
from collections import defaultdict # Use defaultdict for cleaner grouping
import math # For float validation

# Define class IDs and names for clarity
TARGET_CLASSES = {
    0: "human",
    17: "horse"
}
TARGET_CLASS_IDS = set(TARGET_CLASSES.keys()) # Use a set for efficient lookup

# --- Helper function (keep if needed, but focus on main logic first) ---
# def is_finite_coords(coords_list): ...

def process_video_segmentation_batch_efficient(video_path, batch_size=32, output_dir='data_json_output_by_frame', model_name='yolo11x-seg.pt', conf_threshold=0.4, iou_threshold=0.5, target_class_ids=TARGET_CLASS_IDS, frame_skip_rate=3):
    """
    Processes a single video for segmentation efficiently using batched inference,
    reading frames sequentially to conserve memory.
    Detects specified classes and groups results by frame.
    Includes validation for non-finite numbers and ensures JSON serializable types.

    Args:
        video_path (str): Path to the input video file.
        batch_size (int): Batch size for YOLO inference. Adjust based on GPU memory.
        output_dir (str): Base directory to save the JSON output relative to input structure.
        model_name (str): Name of the YOLO segmentation model file.
        conf_threshold (float): Confidence threshold for detections.
        iou_threshold (float): IoU threshold for NMS.
        target_class_ids (set): A set of class IDs to detect.
        frame_skip_rate (int): Process every nth frame.
    """
    # --- Output Path Setup ---
    # json_path = None
    # Generate output path relative to the input structure under output_dir
    # try:
        # This assumes 'dataset' is the root you want to mirror under output_dir
        # Example: /home/ubuntu/workspace/dataset/AUS/vid.mp4 -> /home/ubuntu/workspace/data_cropped_new/AUS/vid_seg.json
        # relative_path_from_root = os.path.relpath(os.path.dirname(video_path), start=input_video_root) # Use global input_video_root
        # output_subdir = os.path.join(output_dir, relative_path_from_root)
        # os.makedirs(output_subdir, exist_ok=True)

        # base_filename = os.path.basename(video_path)
        # json_filename = os.path.splitext(base_filename)[0] + '_seg.json'
        # json_path = os.path.join(output_subdir, json_filename)


    json_path = video_path.replace('dataset', 'data_cropped_new').replace('.mp4', '_seg.json')
    dir_name = os.path.dirname(video_path).replace('dataset', 'data_cropped_new')
    os.makedirs(dir_name, exist_ok=True)
                

        # # --- Skip if JSON already exists ---
        # if os.path.exists(json_path):
        #     print(f"Skipping {video_path}, JSON already exists: {json_path}")
        #     return # Exit the function if output already exists

    # except Exception as path_e:
    #     print(f"\nError determining output path for {video_path}: {path_e}")
    #     traceback.print_exc()
    #     # Log error to a central file
    #     error_log_path = 'seg_path_error.txt'
    #     try:
    #         with open(error_log_path, 'a') as ef: # Append mode
    #            ef.write(f"Path Error for video {video_path}:\n{path_e}\n")
    #            traceback.print_exc(file=ef)
    #            ef.write("-" * 20 + "\n")
    #     except Exception as log_e:
    #         print(f"Additionally, failed to write path error log: {log_e}")
    #     return # Stop processing this video if path fails

    # --- Model Loading ---
    # It might be slightly more efficient to load the model once per process
    # instead of once per video, but this way is simpler and ensures isolation.
    model = None
    cap = None
    try:
        model = YOLO(model_name, verbose=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # --- Video Reading ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            # Create an empty JSON file to mark as processed (or indicate error)
            try:
                with open(json_path, 'w') as f: json.dump({}, f)
            except Exception as empty_json_e:
                 print(f"Error creating empty JSON for failed video {video_path}: {empty_json_e}")
            return

        frame_detections_map = defaultdict(list)
        frames_batch = []
        batch_indices = []
        all_processed_frame_indices = set() # Keep track of indices we *should* have entries for
        frame_idx = 0
        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Get total for progress (optional)

        # --- Sequential Frame Reading and Batched Inference ---
        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of video

            current_frame_idx = frame_idx
            frame_idx += 1

            if current_frame_idx % frame_skip_rate == 0:
                all_processed_frame_indices.add(str(current_frame_idx)) # Record frame index
                frames_batch.append(frame)
                batch_indices.append(current_frame_idx)

                # When batch is full, process it
                if len(frames_batch) == batch_size:
                    results = model.predict(frames_batch,
                                            conf=conf_threshold,
                                            iou=iou_threshold,
                                            verbose=False, stream=False, imgsz=640, device=device)

                    # Process results for this batch
                    for i, result in enumerate(results):
                        original_frame_idx = batch_indices[i]
                        frame_key = str(original_frame_idx)
                        # Access frame_key to ensure it exists in the defaultdict
                        current_frame_list = frame_detections_map[frame_key]

                        if result.boxes is None or result.masks is None:
                            continue # No detections or masks in this frame of the batch

                        # --- Extract data (same logic as before) ---
                        boxes = result.boxes
                        masks = result.masks
                        # Handle cases where masks.xy might be None even if masks exist
                        polygons = masks.xy if masks is not None and masks.xy is not None else []
                        scores = boxes.conf.cpu().numpy().tolist() # Ensure numpy->list conversion
                        class_ids = list(map(int, boxes.cls.cpu().numpy().tolist()))

                        for det_idx, detected_class_id in enumerate(class_ids):
                            if detected_class_id in target_class_ids:
                                conf_score = float(scores[det_idx])
                                segmentation_data = polygons[det_idx].tolist() if det_idx < len(polygons) else None
                                bbox_int_list = list(map(int, boxes.xyxy[det_idx].cpu().numpy().tolist()))
                                detection_data = {
                                    "class_id": detected_class_id,
                                    "class_name": TARGET_CLASSES.get(detected_class_id, "unknown"),
                                    "confidence": conf_score,
                                    "bbox": bbox_int_list,
                                    "segmentation": segmentation_data
                                }
                                current_frame_list.append(detection_data)
                    # --- End processing batch results ---

                    # Clear the batch for the next iteration
                    frames_batch.clear()
                    batch_indices.clear()

        # --- Process the last partial batch (if any) ---
        if frames_batch:
            results = model.predict(frames_batch,
                                    conf=conf_threshold,
                                    iou=iou_threshold,
                                    verbose=False, stream=False, imgsz=640, device=device)
            # Process results (same logic as above)
            for i, result in enumerate(results):
                original_frame_idx = batch_indices[i]
                frame_key = str(original_frame_idx)
                current_frame_list = frame_detections_map[frame_key] # Ensures key exists

                if result.boxes is None or result.masks is None:
                    continue

                boxes = result.boxes
                masks = result.masks
                polygons = masks.xy if masks is not None and masks.xy is not None else []
                scores = boxes.conf.cpu().numpy().tolist()
                class_ids = list(map(int, boxes.cls.cpu().numpy().tolist()))

                for det_idx, detected_class_id in enumerate(class_ids):
                     if detected_class_id in target_class_ids:
                        conf_score = float(scores[det_idx])
                        segmentation_data = polygons[det_idx].tolist() if det_idx < len(polygons) else None
                        bbox_int_list = list(map(int, boxes.xyxy[det_idx].cpu().numpy().tolist()))

                        detection_data = {
                            "class_id": detected_class_id,
                            "class_name": TARGET_CLASSES.get(detected_class_id, "unknown"),
                            "confidence": conf_score,
                            "bbox": bbox_int_list,
                            "segmentation": segmentation_data
                        }
                        current_frame_list.append(detection_data)
            # No need to clear batch here, loop is ending

        # --- Ensure all processed frames have an entry in the map ---
        for frame_key_str in all_processed_frame_indices:
            if frame_key_str not in frame_detections_map:
                frame_detections_map[frame_key_str] = [] # Add empty list for frames with no detections

        # --- Save Results to JSON ---
        # num_detected_frames = len(frame_detections_map)
        # total_detections = sum(len(v) for v in frame_detections_map.values())
        # print(f"Writing JSON for {video_path}: {total_detections} detections across {num_detected_frames} frames to {json_path}")

        with open(json_path, 'w') as f:
            sorted_frame_keys = sorted(frame_detections_map.keys(), key=int)
            sorted_frame_detections_map = {k: frame_detections_map[k] for k in sorted_frame_keys}
            try:
                json.dump(sorted_frame_detections_map, f, indent=4, allow_nan=False)
            except ValueError as json_err:
                 print(f"\nFATAL ERROR during JSON dump for {video_path}: {json_err}")
                 print("This likely means non-finite numbers (NaN/Inf) were still present despite validation checks.")
                 # Clean up potentially corrupted file
                 f.close() # Close file before attempting removal
                 if os.path.exists(json_path):
                    try:
                        os.remove(json_path)
                        print(f"Removed potentially corrupted file: {json_path}")
                    except OSError as remove_err:
                        print(f"Error removing corrupted file {json_path}: {remove_err}")
                 return # Stop processing this video

    except Exception as e:
        print(f"\nError processing video {video_path}: {e}")
        traceback.print_exc()
        # Log error to a central file
        error_log_path = 'seg_processing_error.txt'
        try:
            with open(error_log_path, 'a') as ef: # Append mode
               ef.write(f"Error processing video {video_path}:\n{e}\n")
               traceback.print_exc(file=ef)
               ef.write("-" * 20 + "\n")
        except Exception as log_e:
            print(f"Additionally, failed to write processing error log: {log_e}")
        # Attempt to remove potentially incomplete JSON if an error occurred mid-processing
        if json_path and os.path.exists(json_path):
             try:
                 os.remove(json_path)
                 print(f"Removed potentially incomplete JSON due to error: {json_path}")
             except OSError as remove_err:
                 print(f"Error removing incomplete JSON {json_path}: {remove_err}")


    finally:
        # --- Cleanup ---
        if cap is not None and cap.isOpened():
            cap.release()
        # Unload model and clear GPU memory (important in multiprocessing)
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --- Multiprocessing Wrappers ---
# (Keep these as they are)
def process_video_wrapper(video_path, batch_size, output_dir, model_name, conf_threshold, iou_threshold, target_class_ids, frame_skip_rate):
    """Wrapper function"""
    # Pass the global input_video_root implicitly, or explicitly add it as an argument if preferred
    process_video_segmentation_batch_efficient(video_path, batch_size, output_dir, model_name, conf_threshold, iou_threshold, target_class_ids, frame_skip_rate)

def process_video_wrapper_star(args):
    """Helper function to unpack arguments"""
    return process_video_wrapper(*args)

# --- File Listing ---
def list_mp4_files(root_dir, output_base_dir):
    """Recursively finds MP4 files, excluding common unwanted ones and ones already processed."""
    mp4_files = []
    print(f"Searching for MP4 files in: {root_dir}")
    if not os.path.isdir(root_dir):
        print(f"Error: Input directory '{root_dir}' not found.")
        return []

    skipped_count = 0
    found_count = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Pruning: Skip known non-video directories if applicable
        # dirnames[:] = [d for d in dirnames if d not in {'__pycache__', '.git'}]

        for filename in filenames:
            lower_filename = filename.lower()
            # Basic MP4 check
            if not lower_filename.endswith('.mp4'):
                continue
            # Exclude specific patterns
            if "_vis.mp4" in lower_filename or \
               'dlc_resnet50' in lower_filename or \
               'labeled.mp4' in lower_filename:
                continue

            file_path = os.path.join(dirpath, filename)

            if not os.path.exists(file_path.replace('dataset', 'data_cropped_new').replace('.mp4', '_seg.json')):
                mp4_files.append(file_path)
                found_count += 1
            else:
                skipped_count += 1
                continue # Skip this file, output exists


    print(f"Found {found_count} new MP4 files to process.")
    print(f"Skipped {skipped_count} files where output JSON already exists.")
    return mp4_files

# Define input root globally or pass it around
input_video_root = '/home/ubuntu/workspace/dataset'
output_base_dir = "/home/ubuntu/workspace/data_cropped_new" # Output directory name

# --- Main Execution Block ---
if __name__ == "__main__":
    # Set input and output directories (defined globally above)

    # Get list of videos to process
    # Pass output_base_dir to list_mp4_files for checking existing outputs
    video_list = sorted(list_mp4_files(input_video_root, output_base_dir))

    if not video_list:
        print("No new videos found to process. Exiting.")
        exit()

    # --- Configuration ---
    batch_size = 48             # *** TRY REDUCING BATCH SIZE FIRST ***
    model_name = 'yolo11x-seg.pt'
    conf_threshold = 0.5     # Adjusted confidence slightly
    iou_threshold = 0.5
    frame_skip_rate = 3

    # --- Multiprocessing Setup ---
    # *** TRY REDUCING PROCESS COUNT FIRST ***
    # Start with fewer processes than CPU cores, especially if memory is tight
    num_processes = min(mp.cpu_count(), 6) # Try 4 or 6 first, not max
    # num_processes = 1 # Use 1 for debugging to isolate issues

    print(f"\n--- Starting Video Processing ---")
    print(f"Input video root: {input_video_root}")
    print(f"Output JSON directory: {output_base_dir}")
    print(f"Number of NEW videos to process: {len(video_list)}")
    print(f"Using {num_processes} parallel processes.")
    print(f"Configuration: Batch Size={batch_size}, Model={model_name}, Conf={conf_threshold}, Skip Rate={frame_skip_rate}")
    print(f"Target Classes: {TARGET_CLASSES}")
    print(f"Processing Mode: Sequential frame reading per video (Memory Optimized)")
    print(f"---------------------------------\n")

    # Prepare arguments for mapping
    args_list = [
        (video_file, batch_size, output_base_dir, model_name, conf_threshold, iou_threshold, TARGET_CLASS_IDS, frame_skip_rate)
        for video_file in video_list
    ]

    # Set start method (optional but can help on some systems)
    try:
        # 'spawn' is generally safer across platforms than 'fork' when using CUDA/GPU resources
        mp.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'")
    except RuntimeError:
        print("Multiprocessing start method already set or 'spawn' not required/supported.")
        pass

    # Use multiprocessing pool
    with mp.Pool(processes=num_processes) as pool:
        # Use imap_unordered for potentially better load balancing
        # Use tqdm for progress bar
        results = list(tqdm(
            pool.imap_unordered(process_video_wrapper_star, args_list),
            total=len(video_list),
            desc="Overall Progress"
        ))

    print("\n--- Video processing complete. ---")
    print("Check seg_processing_error.txt and seg_path_error.txt for any logged errors.")