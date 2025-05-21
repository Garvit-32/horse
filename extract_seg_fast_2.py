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

# --- Helper function to check for NaN/Infinity in coordinate lists ---
# (Keep this from the previous fix, it's still good practice)
# def is_finite_coords(coords_list):
#     """Checks if all numbers in a list or list-of-lists are finite."""
#     if coords_list is None:
#         return True
#     if not isinstance(coords_list, (list, tuple)):
#         return False
#     for item in coords_list:
#         if isinstance(item, (list, tuple)):
#             for val in item:
#                 if not isinstance(val, (int, float)) or not math.isfinite(val):
#                     return False
#         elif isinstance(item, (int, float)):
#             if not math.isfinite(item):
#                 return False
#         else:
#             return False
#     return True

def process_video_segmentation_batch_efficient(video_path, batch_size=32, output_dir='data_json_output_by_frame', model_name='yolo11x-seg.pt', conf_threshold=0.4, iou_threshold=0.5, target_class_ids=TARGET_CLASS_IDS, frame_skip_rate=3):
    """
    Processes a single video for segmentation efficiently using batched inference,
    detecting specified classes (human and horse by default) and grouping results by frame.
    Includes validation for non-finite numbers and ensures JSON serializable types.

    Args:
        video_path (str): Path to the input video file.
        batch_size (int): Batch size for YOLO inference. Adjust based on GPU memory.
        output_dir (str): Directory to save the JSON output.
        model_name (str): Name of the YOLO segmentation model file.
        conf_threshold (float): Confidence threshold for detections.
        iou_threshold (float): IoU threshold for NMS.
        target_class_ids (set): A set of class IDs to detect (default: {0 for human, 17 for horse}).
        frame_skip_rate (int): Process every nth frame (e.g., 3 for every 3rd frame).
    """
    # --- Output Path Setup ---
    json_path = None
    try:
        # try:
        #     relative_path = os.path.relpath(os.path.dirname(video_path), start=input_video_root)
        # except ValueError:
        #     relative_path = ""
        # output_subdir = os.path.join(output_dir, relative_path)
        # os.makedirs(output_subdir, exist_ok=True)
        # base_filename = os.path.basename(video_path)
        # json_filename = os.path.splitext(base_filename)[0] + '_seg.json'
        # json_path = os.path.join(output_subdir, json_filename)
        json_path = video_path.replace('dataset', 'data_cropped_new').replace('.mp4', '_seg.json')
        dir_name = os.path.dirname(video_path).replace('dataset', 'data_cropped_new')
        os.makedirs(dir_name, exist_ok=True)
                

        # --- Model Loading ---
        model = YOLO(model_name, verbose=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # --- Video Reading ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            if json_path:
                with open(json_path, 'w') as f: json.dump({}, f)
            return

        # --- Frame Reading and Batching ---
        frame_detections_map = defaultdict(list)
        frames_batches = []
        frames_idxs = []
        processed_frame_indices = set() # Keep track for ensuring all frames are in output
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip_rate == 0:
                frames_batches.append(frame)
                frames_idxs.append(frame_idx)
                processed_frame_indices.add(str(frame_idx))

            frame_idx += 1

        cap.release()

        num_frames_to_process = len(frames_batches)
        if not frames_batches:
            print(f"No frames selected for processing in {video_path}.")
            if json_path:
                 # Ensure even empty JSON has keys for all processed frames
                 empty_data = {idx_str: [] for idx_str in sorted(list(processed_frame_indices), key=int)}
                 with open(json_path, 'w') as f: json.dump(empty_data, f, indent=4)
            return

        # --- Batched Inference and Result Processing ---
        # print(f"Processing {num_frames_to_process} frames for {os.path.basename(video_path)}...")

        for offset in range(0, num_frames_to_process, batch_size):
            batch_indices = frames_idxs[offset:offset + batch_size]
            frames_batch = frames_batches[offset:offset + batch_size]

            results = model.predict(frames_batch,
                                    conf=conf_threshold,
                                    iou=iou_threshold,
                                    verbose=False, stream=False, imgsz=640, device=device)

            for frame_idx_in_batch, result in enumerate(results):
                current_frame_original_idx = batch_indices[frame_idx_in_batch]
                frame_key = str(current_frame_original_idx)

                # Access frame_key to ensure it exists in the defaultdict
                current_frame_list = frame_detections_map[frame_key]

                if result.boxes is None or result.masks is None:
                    continue

                boxes = result.boxes
                masks = result.masks
                polygons = masks.xy if masks.xy is not None else []
                scores = boxes.conf.tolist() # Should be list of Python floats
                class_ids =  list(map(int, boxes.cls.cpu().numpy().tolist()))

                # Iterate through detections in the current frame
                for i, detected_class_id in enumerate(class_ids): # Keep original name for clarity
        
                    if detected_class_id in target_class_ids:
                        # --- Data Extraction and Validation ---
                        # valid_detection = True
                        # bbox_data_raw =  # list of floats
                        # Ensure confidence is Python float
                        conf_score = float(scores[i])
                        segmentation_data = polygons[i].tolist() if i < len(polygons) else []
                        # segmentation_data_raw = polygons[i].tolist()

                        # Extract segmentation
                        # if i < len(polygons) and polygons[i] is not None:
                            # Convert points to Python floats within the list comprehension
                            
                            # Validate segmentation points before final assignment
                        #     if is_finite_coords(segmentation_data_raw):
                        #         segmentation_data = [[float(p[0]), float(p[1])] for p in segmentation_data_raw]
                        #     else:
                        #         print(f"\nWarning: Skipping detection (frame {frame_key}, class {detected_class_id}) due to non-finite segmentation points.")
                        #         valid_detection = False
                        # # If seg is missing, segmentation_data remains []

                        # # Validate BBox coordinates
                        # if not is_finite_coords(bbox_data_raw):
                        #     print(f"\nWarning: Skipping detection (frame {frame_key}, class {detected_class_id}) due to non-finite bbox: {bbox_data_raw}")
                        #     valid_detection = False

                        # # Validate Confidence score
                        # if not math.isfinite(conf_score):
                        #     print(f"\nWarning: Skipping detection (frame {frame_key}, class {detected_class_id}) due to non-finite confidence: {conf_score}")
                        #     valid_detection = False

                        # --- Append only if VALID ---
                    # if valid_detection:
                        # Convert bbox floats to Python ints AFTER validation
                        bbox_int_list = list(map(int, boxes.xyxy[i].tolist()))

                        # Create dict using validated and correctly typed data
                        detection_data = {
                            "class_id": detected_class_id,  # Already Python int
                            "class_name": TARGET_CLASSES.get(detected_class_id, "unknown"),
                            "confidence": conf_score,      # Already Python float
                            "bbox": bbox_int_list,         # Already list of Python ints
                            "segmentation": segmentation_data # List of lists of Python floats, or []
                        }
                        # Append this detection to the list for the current frame index
                        current_frame_list.append(detection_data)
                        # --- End Append only if VALID ---

        # --- Ensure all processed frames have an entry in the map ---
        for frame_key in processed_frame_indices:
            if frame_key not in frame_detections_map:
                frame_detections_map[frame_key] = [] # Add empty list for frames with no detections

        # --- Save Results to JSON ---
        num_detected_frames = len(frame_detections_map)
        total_detections = sum(len(v) for v in frame_detections_map.values())
        # print(f"Writing JSON for {video_path}: {total_detections} detections across {num_detected_frames} frames to {json_path}")
        with open(json_path, 'w') as f:
            # Sort keys numerically for ordered output
            sorted_frame_keys = sorted(frame_detections_map.keys(), key=int)
            sorted_frame_detections_map = {k: frame_detections_map[k] for k in sorted_frame_keys}
            try:
                # Use allow_nan=False as a final safeguard against errors
                json.dump(sorted_frame_detections_map, f, indent=4, allow_nan=False)
            except ValueError as json_err:
                 print(f"\nFATAL ERROR during JSON dump for {video_path}: {json_err}")
                 print("This likely means non-finite numbers (NaN/Inf) were still present despite validation checks.")
                 f.close()
                 if os.path.exists(json_path): os.remove(json_path)
                 print(f"Removed potentially corrupted file: {json_path}")
                 # Optional: re-raise error if needed for multiprocessing management
                 # raise json_err
                 return # Stop processing this video

    except Exception as e:
        print(f"\nError processing video {video_path}: {e}")
        traceback.print_exc()
        if json_path: # Only try to write error file if path was determined
            error_log_path = 'seg_error.txt'
            try: # Prevent error during error logging
                with open(error_log_path, 'w') as ef:
                   ef.write(f"Error processing video {video_path}:\n{e}\n")
                   traceback.print_exc(file=ef)
            except Exception as log_e:
                print(f"Additionally, failed to write error log: {log_e}")


# --- Multiprocessing Wrappers ---
def process_video_wrapper(video_path, batch_size, output_dir, model_name, conf_threshold, iou_threshold, target_class_ids, frame_skip_rate):
    """Wrapper function"""
    process_video_segmentation_batch_efficient(video_path, batch_size, output_dir, model_name, conf_threshold, iou_threshold, target_class_ids, frame_skip_rate)

def process_video_wrapper_star(args):
    """Helper function to unpack arguments"""
    return process_video_wrapper(*args)

# --- File Listing ---
def list_mp4_files(root_dir):
    """Recursively finds MP4 files, excluding common unwanted ones."""
    mp4_files = []
    print(f"Searching for MP4 files in: {root_dir}")
    if not os.path.isdir(root_dir):
        print(f"Error: Input directory '{root_dir}' not found.")
        return []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            lower_filename = filename.lower()
            if lower_filename.endswith('.mp4') and \
               "_vis.mp4" not in lower_filename and \
               'dlc_resnet50' not in lower_filename and \
               'labeled.mp4' not in lower_filename:
                file_path = os.path.join(dirpath, filename)
                
                if not os.path.exists(file_path.replace('dataset', 'data_cropped_new').replace('.mp4', '_seg.json')):
                    mp4_files.append(file_path)
    print(f"Found {len(mp4_files)} MP4 files.")
    return mp4_files

# /home/ubuntu/workspace/data_cropped_new/AUS/sale_2024_australian_easter_yearling_sale/Lot 20_ Per Incanto (USA)_Farrelly/video_20_seg.json

# /home/ubuntu/workspace/dataset/AUS/sale_2024_australian_easter_yearling_sale/Lot 20_ Per Incanto (USA)_Farrelly/video_20.mp4

input_video_root = '/home/ubuntu/workspace/dataset'
# --- Main Execution Block ---
if __name__ == "__main__":
    # Set input and output directories
    output_base_dir = "data_cropped_new" # Output directory name

    # Get list of videos to process
    # video_list = list_mp4_files(input_video_root)[:10] # Process only first 10 for testing
    # video_list = sorted(list_mp4_files(input_video_root)) # Process all found videos
    


    json_list = [r'/home/ubuntu/workspace/data_cropped/tattersales_ireland/sales_september-yearling-sale-part-2_4DCGI_Sale_SY223_Main/Lot 633 Inns of Court (IRE) _ Lon Dubh (IRE) B.F. (IRE)/video_1_1_lot%20633%20complete_seg.json', '/home/ubuntu/workspace/data_cropped/tattersales_ireland/sales_winter-flat-and-national-hunt-sale_4DCGI_Sale_FNH19_Main/Lot 326 Pakora (FR) Gr.M/video_2_1_1601e9cd-4bdd-4afb-8afc-08f9ad2ccda9_seg.json', '/home/ubuntu/workspace/data_cropped/tattertailsv1/4DCGI_Sale_DEM 20/Lot 2235 - Tilia Cordata_cropped (IRE) B.F. BACK TO LIST/video_1_1_Tilliaa_seg.json', '/home/ubuntu/workspace/data_cropped/fastiptondata/2021_Kentucky-Winter-Mixed/E Z KITTY_lot_587/video_seg.json', '/home/ubuntu/workspace/data_cropped/fastiptondata/2020_The-November-Sale/STORM THE HILL_lot_215/video_seg.json', '/home/ubuntu/workspace/data_cropped/fastiptondata/2020_The-November-Sale/MOTHER MOTHER_lot_193/video_seg.json', '/home/ubuntu/workspace/data_cropped/fastiptondata/2020_The-November-Sale/MY GIRL RED_lot_184/video_seg.json', '/home/ubuntu/workspace/data_cropped/fastiptondata/2020_The-November-Sale/Unnamed_Horse_89_lot_89/video_seg.json', '/home/ubuntu/workspace/data_cropped/fastiptondata/2020_The-November-Sale/Unnamed_Horse_75_lot_75/video_seg.json', '/home/ubuntu/workspace/data_cropped/fastiptondata/2020_The-November-Sale/Unnamed_Horse_148_lot_148/video_seg.json', '/home/ubuntu/workspace/data_cropped/fastiptondata/2020_The-November-Sale/Unnamed_Horse_80_lot_80/video_seg.json', '/home/ubuntu/workspace/data_cropped/fastiptondata/2020_The-November-Sale/Unnamed_Horse_2_lot_2/video_seg.json', '/home/ubuntu/workspace/data_cropped/fastiptondata/2020_The-November-Sale/Unnamed_Horse_99_lot_99/video_seg.json', '/home/ubuntu/workspace/data_cropped/fastiptondata/2020_The-November-Sale/CLASSY ACT_lot_258/video_seg.json', '/home/ubuntu/workspace/data_cropped/fastiptondata/2020_The-November-Sale/Unnamed_Horse_108_lot_108/video_seg.json', '/home/ubuntu/workspace/data_cropped/fastiptondata/2020_The-November-Sale/Unnamed_Horse_13_lot_13/video_seg.json', '/home/ubuntu/workspace/data_cropped/fastiptondata/2020_The-November-Sale/Unnamed_Horse_52_lot_52/video_seg.json', '/home/ubuntu/workspace/data_cropped/fastiptondata/2020_The-November-Sale/RARE DAY (IRE)_lot_201/video_seg.json', '/home/ubuntu/workspace/data_cropped/fastiptondata/2020_The-November-Sale/POSITIVE SPIRIT_lot_226/video_seg.json', '/home/ubuntu/workspace/data_cropped/fastiptondata/2020_The-November-Sale/Unnamed_Horse_136_lot_136/video_seg.json', '/home/ubuntu/workspace/data_cropped/fastiptondata/2020_The-November-Sale/ULTIMA D_lot_224/video_seg.json', '/home/ubuntu/workspace/data_cropped/fastiptondata/2020_The-November-Sale/COMPETITIONOFIDEAS_lot_259/video_seg.json', '/home/ubuntu/workspace/data_cropped/fastiptondata/2020_The-November-Sale/Unnamed_Horse_7_lot_7/video_seg.json', '/home/ubuntu/workspace/data_cropped/fastiptondata/2020_The-November-Sale/Unnamed_Horse_40_lot_40/video_seg.json', '/home/ubuntu/workspace/data_cropped/fastiptondata/2020_The-November-Sale/Unnamed_Horse_84_lot_84/video_seg.json']
    

    video_list = [i.replace('data_cropped', 'dataset').replace('_seg.json', '.mp4') for i in json_list]




    # if not video_list:
    #     print("No videos found to process. Exiting.")
    #     exit()

    # --- Configuration ---
    batch_size = 32             # Adjust based on GPU memory
    model_name = 'yolo11x-seg.pt' # Make sure this model file exists or change it
    conf_threshold = 0.5     # Detection confidence threshold
    iou_threshold = 0.5         # NMS IoU threshold
    frame_skip_rate = 3         # Process every Nth frame

    # --- Multiprocessing Setup ---
    num_processes = min(mp.cpu_count(), 6) # Limit processes
    # num_processes = 1 # Use 1 for debugging

    print(f"\n--- Starting Video Processing ---")
    print(f"Input video root: {input_video_root}")
    print(f"Output JSON directory: {output_base_dir}")
    print(f"Number of videos to process: {len(video_list)}")
    print(f"Using {num_processes} parallel processes.")
    print(f"Configuration: Batch Size={batch_size}, Model={model_name}, Conf={conf_threshold}, Skip Rate={frame_skip_rate}")
    print(f"Target Classes: {TARGET_CLASSES}")
    print(f"JSON Validation: Enabled (skipping NaN/Infinity, ensuring Python types)")
    print(f"---------------------------------\n")


    # Prepare arguments for mapping
    args_list = [
        (video_file, batch_size, output_base_dir, model_name, conf_threshold, iou_threshold, TARGET_CLASS_IDS, frame_skip_rate)
        for video_file in video_list
    ]

    # Use multiprocessing pool
    # Optional: set start method if needed
    # try:
    #     mp.set_start_method('spawn', force=True)
    # except RuntimeError:
    #     pass

    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_video_wrapper_star, args_list),
            total=len(video_list),
            desc="Overall Progress"
        ))

    print("\n--- Video processing complete. ---")