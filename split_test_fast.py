import argparse
import os
import sys
import json
import cv2
import numpy as np
import torch
import pickle
from torchvision import transforms
from tqdm import tqdm
import collections
from statistics import mode, StatisticsError
import torch.nn.functional as F
from PIL import Image
import functools
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import time  # For basic profiling if needed

import warnings
warnings.filterwarnings("ignore")

# Enable cudnn benchmark for faster runtime (if input sizes are fixed)
torch.backends.cudnn.benchmark = True

# Normalization transform (used for DINO)
dino_transform = transforms.Compose([
    transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Global variables to be set by worker_init_fn
# Using Manager list for process-safe error reporting
MANAGER = None
GLOBAL_MODEL = None
ARGS = None
PROTOTYPES = None
GLOBAL_ERROR_LIST = None # This will be a Manager.list

def parse_args():
    parser = argparse.ArgumentParser(description="Optimized Video Processing Script")
    # --- Essential Paths ---
    parser.add_argument('--data_dir', type=str, default = '/workspace/Dessie',  help="Root directory containing 'data_tensor', 'data', 'data_cropped'")
    parser.add_argument('--save_dir', type=str, default = 'videos',  help="Root directory to save processed videos")
    parser.add_argument('--prototype_path', type=str, default='/workspace/Dessie/prototypes.pth', help="Path to pre-computed DINO prototypes")
    parser.add_argument('--ckpt_file', type=str, help="Path to DESSIE checkpoint (if used, currently inactive)") # Kept if needed later

    # --- Processing Parameters ---
    parser.add_argument('--num_workers', type=int, default=8, help="Number of worker processes for DataLoader")
    parser.add_argument('--dino_batch_size', type=int, default=2, help="Batch size for DINO inference")
    parser.add_argument('--imgsize', type=int, default=256, help="Target size for cropping/resizing (before DINO)") # Keep if used elsewhere
    parser.add_argument('--output_fps', type=int, default=20, help="FPS for the output videos")
    parser.add_argument('--orientation_conf', type=float, default=0.5, help="Confidence threshold for kp-based orientation dx vs dz")
    parser.add_argument('--movement_threshold', type=float, default=1.5, help="Avg keypoint movement threshold (pixels) to determine if horse is moving")
    parser.add_argument('--smoothing_window', type=int, default=5, help="Window size for temporal smoothing of orientation")
    parser.add_argument('--bbox_expand_ratio', type=float, default=0.05, help="Ratio to expand bounding boxes")

    # --- Model & Feature Parameters ---
    parser.add_argument("--dino_model_name", type=str, default="dinov2_vitb14", help="DINOv2 model name")

    # --- Debugging/Testing ---
    parser.add_argument('--limit_videos', type=int, default=None, help="Process only the first N videos for testing")
    parser.add_argument('--error_log_file', type=str, default='processing_errors.pkl', help="File to save list of errored video paths")

    # --- Arguments kept from original but potentially unused with current DINO focus ---
    # These are kept for potential future use or if parts of DESSIE logic are re-enabled
    parser.add_argument('--train', type=str)
    parser.add_argument('--model_dir', type=str, default='/home/ubuntu/workspace/Dessie/internal_models') # Example path
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--version', type=str)
    parser.add_argument('--ModelName', type=str, default='DESSIE') # Inactive
    parser.add_argument('--DatasetName', type=str, default='DessiePIPE') # Inactive
    parser.add_argument('--batch_size', type=int, default=1, help="DataLoader batch size (usually 1 for video processing)") # Renamed for clarity
    parser.add_argument('--W_shape_prior', default=50., type=float) # Inactive weights
    parser.add_argument('--W_kp_img', default=0.001, type=float)
    parser.add_argument('--W_mask_img', default=0.0001, type=float)
    parser.add_argument('--W_pose_img', default=0.01, type=float)
    # ... other weights ...
    parser.add_argument('--lr', type=float, default=5e-05)
    parser.add_argument('--max-epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--PosePath', type=str)
    parser.add_argument("--TEXTUREPath", type=str)
    parser.add_argument('--uv_size', type=int, default=256)
    parser.add_argument('--data_batch_size', type=int, default=1) # Potentially confusing name, keep if needed
    parser.add_argument('--useSynData', action="store_true")
    parser.add_argument('--useinterval', type=int, default=8)
    parser.add_argument("--getPairs", action="store_true", default=False)
    parser.add_argument("--TEXT", action="store_true", default=False)
    parser.add_argument("--DINO_frozen", action="store_true", default=False)
    parser.add_argument("--DINO_obtain_token", action="store_true", default=False)
    parser.add_argument("--GT", action="store_true", default=False)
    parser.add_argument('--W_gt_shape', default=0, type=float)
    parser.add_argument('--W_gt_pose', default=0., type=float)
    parser.add_argument('--W_gt_trans', default=0., type=float)
    parser.add_argument("--pred_trans", action="store_true", default=False)
    parser.add_argument("--background", action="store_true", default=False)
    parser.add_argument("--background_path", default='')
    parser.add_argument("--REALDATASET", default='MagicPony')
    parser.add_argument("--REALPATH", default='')
    parser.add_argument("--web_images_num", type=int, default=0)
    parser.add_argument("--REALMagicPonyPATH", default='')

    return parser.parse_args()

# Removed set_default_args as defaults are now in argparse


def convert_3d_to_2d(points, size=256):
    # Keep this function if pred_data["pred_kp3d_crop"] is used
    focal = 5000
    R = np.array([[-1, 0, 0],
                  [0, -1, 0],
                  [0, 0, 1]])
    # Ensure points is numpy array on CPU
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    points_cam = points @ R.T
    x_cam = points_cam[:, 0]
    y_cam = points_cam[:, 1]
    z_cam = points_cam[:, 2]

    # Avoid division by zero or very small numbers
    z_cam[z_cam < 1e-6] = 1e-6

    x_proj = (focal * x_cam) / z_cam
    y_proj = (focal * y_cam) / z_cam
    screen_x = (size / 2) + x_proj
    screen_y = (size / 2) - y_proj
    proj_points = np.stack([screen_x, screen_y], axis=-1)
    return proj_points.astype(np.int32)


def list_pt_files(root_dir):
    """Lists all .pt files recursively in the root_dir."""
    files = []
    if not os.path.isdir(root_dir):
        print(f"Error: Tensor directory not found: {root_dir}")
        return files
    for dp, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith('.pt'):
                files.append(os.path.join(dp, f))
    return files


def expand_bbox(bbox, frame_width, frame_height, expand_ratio=0.05):
    x1, y1, x2, y2 = map(int, bbox) # Ensure integer coordinates
    w_box = x2 - x1
    h_box = y2 - y1
    ex_w = int(w_box * expand_ratio)
    ex_h = int(h_box * expand_ratio)
    x1_new = max(0, x1 - ex_w)
    y1_new = max(0, y1 - ex_h)
    x2_new = min(frame_width - 1, x2 + ex_w) # Use frame_width-1 and frame_height-1
    y2_new = min(frame_height - 1, y2 + ex_h)
    return [x1_new, y1_new, x2_new, y2_new]


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


def get_dino_embedding_batch(image_batch_tensor, model):
    """Processes a batch of image tensors with DINO."""
    device = next(model.parameters()).device
    image_batch_tensor = image_batch_tensor.to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        embeddings = model(image_batch_tensor)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings # Returns embeddings on GPU

def predict_orientation_dino_batch(embeddings_batch, prototypes):
    """Calculates similarity scores for a batch of embeddings against prototypes."""
    orientations = list(prototypes.keys())
    proto_tensor = torch.cat([prototypes[o] for o in orientations], dim=0).to(embeddings_batch.device) # Shape: [num_orientations, embed_dim]

    # Calculate cosine similarity (dot product of normalized vectors)
    # Input: [batch_size, embed_dim], [num_orientations, embed_dim]
    # Output: [batch_size, num_orientations]
    similarity_scores = torch.mm(embeddings_batch, proto_tensor.T)

    # Find the best orientation per sample
    best_scores, best_indices = torch.max(similarity_scores, dim=1)

    results = []
    # Move results to CPU for easier handling later
    similarity_scores_cpu = similarity_scores.cpu().numpy()
    best_indices_cpu = best_indices.cpu().numpy()
    best_scores_cpu = best_scores.cpu().numpy()

    # Generate dictionary output for each item in the batch
    for i in range(embeddings_batch.shape[0]):
        pred_orientation = orientations[best_indices_cpu[i]]
        scores_dict = {orient: score for orient, score in zip(orientations, similarity_scores_cpu[i])}
        
        # Add specific checks like 'occluded' or 'running' if they are part of your prototypes
        occluded = scores_dict.get('occluded', -1) > scores_dict.get('unoccluded', 0) # Example
        running = scores_dict.get('running', -1) > 0.5 # Example threshold

        results.append({
            'orientation': pred_orientation,
            'orientation_score': best_scores_cpu[i],
            'all_scores': scores_dict, # Optional: keep all scores for debugging
            'occluded': occluded, # Example - adapt to your prototypes
            'running': running   # Example - adapt to your prototypes
        })
    return results


def resize_keypoints(keypoints, original_size, new_size):
    """Resizes keypoint coordinates from an original image size to a new image size."""
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.cpu().numpy() # Ensure NumPy array on CPU

    if not isinstance(keypoints, np.ndarray) or keypoints.ndim != 2 or keypoints.shape[1] != 2:
        # Handle case where keypoints might be empty or invalid
        return np.empty((0, 2), dtype=np.int32)
        # raise ValueError("keypoints must be a NumPy array of shape (N, 2)") # Or return empty

    original_height, original_width = original_size
    new_height, new_width = new_size

    if original_width <= 0 or original_height <= 0:
         return np.empty((0, 2), dtype=np.int32) # Or raise error
         # raise ValueError("Original dimensions must be positive.")

    keypoints_float = keypoints.astype(np.float64)

    scale_factors = np.array([
        new_width / original_width if original_width > 0 else 1,
        new_height / original_height if original_height > 0 else 1
    ], dtype=np.float64)

    resized_keypoints = keypoints_float * scale_factors

    # Clamp coordinates to be within the new image bounds
    resized_keypoints[:, 0] = np.clip(resized_keypoints[:, 0], 0, new_width - 1)
    resized_keypoints[:, 1] = np.clip(resized_keypoints[:, 1], 0, new_height - 1)

    return resized_keypoints.astype(np.int32)


def init_worker(worker_id, args_dict, error_list):
    """Initializer for DataLoader workers."""
    global GLOBAL_MODEL, ARGS, PROTOTYPES, GLOBAL_ERROR_LIST
    # print(f"Initializing worker {os.getpid()}...")
    # Use Manager proxy objects within the worker
    # MANAGER = manager_proxy
    GLOBAL_ERROR_LIST = error_list

    ARGS = argparse.Namespace(**args_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Load DINO Model
        GLOBAL_MODEL = torch.hub.load('facebookresearch/dinov2', ARGS.dino_model_name, pretrained=True)
        GLOBAL_MODEL.eval()
        GLOBAL_MODEL.to(device)

        # Load Prototypes
        if not os.path.exists(ARGS.prototype_path):
             raise FileNotFoundError(f"Prototype file not found: {ARGS.prototype_path}")
        PROTOTYPES = torch.load(ARGS.prototype_path, map_location='cpu') # Load to CPU first
        # Move prototypes to GPU (if model is on GPU) - do this once
        if device.type == 'cuda':
             PROTOTYPES = {k: v.to(device) for k, v in PROTOTYPES.items()}

    except Exception as e:
        print(f"!!! FATAL ERROR in worker init {os.getpid()}: {e}")
        # Log the error centrally if possible, or raise to stop the worker
        GLOBAL_ERROR_LIST.append(f"Worker {os.getpid()} failed initialization: {e}")


        import traceback
        traceback.print_exc() # Print traceback for debugging

        raise e # Raising exception might stop the DataLoader pool gracefully

    # print(f"Worker {os.getpid()} initialized successfully.")


def process_video_wrapper(args_tuple):
    """ Unpacks arguments and calls process_video for use with Manager.Pool """
    return process_video(*args_tuple)


def process_video(tensor_file_path: str):
    """
    Processes a single video: loads data, performs batched DINO inference,
    determines orientation frame-by-frame, and saves oriented video clips.
    """
    global GLOBAL_MODEL, ARGS, PROTOTYPES, GLOBAL_ERROR_LIST
    worker_pid = os.getpid()
    # print(f"[Worker {worker_pid}] Processing: {tensor_file_path}")

    # --- 1. Derive Paths ---
    try:
        # base_path = tensor_file_path.replace('data_tensor', '').replace('_results.pt', '')
        # video_path = os.path.join(ARGS.data_dir, 'data', base_path + '.mp4')
        # json_path = os.path.join(ARGS.data_dir, 'data_cropped', base_path + '_seg.json')



        video_path = tensor_file_path.replace('data_tensor', 'data').replace('_results.pt', '.mp4')
        json_path = tensor_file_path.replace('data_tensor', 'data_cropped').replace('_results.pt', '_seg.json')

        # print('='*60)
        # print(video_path, json_path, tensor_file_path)
        # print('='*60)

        # output_base = os.path.join(ARGS.save_dir, base_path)
        # output_dir_name = os.path.dirname(output_base)
        # os.makedirs(output_dir_name, exist_ok=True)


    except Exception as e:
        msg = f"Path derivation error for {tensor_file_path}: {e}"
        print(msg)
        GLOBAL_ERROR_LIST.append(f"[{worker_pid}] {msg}")
        return tensor_file_path # Return path to indicate failure

    # --- 2. Load Input Data ---
    try:
        if not os.path.exists(tensor_file_path): raise FileNotFoundError("Tensor file missing")
        pred_data = torch.load(tensor_file_path, map_location='cpu') # Load to CPU first
        
        if not os.path.exists(json_path): raise FileNotFoundError("JSON file missing")
        with open(json_path, 'r') as f:
            frame_data = json.load(f) # List of dicts {frame_idx, bbox, segmentation}

        if not frame_data:
            # print(f"Warning [{worker_pid}]: No frame entries in JSON {json_path}")
            return None # Success, but nothing to process

    except Exception as e:
        msg = f"Error loading data for {tensor_file_path}: {e}"
        print(msg)
        GLOBAL_ERROR_LIST.append(f"[{worker_pid}] {msg}")
        return tensor_file_path

    # --- 3. Prepare for Frame Processing ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        msg = f"Error opening video {video_path}"
        print(msg)
        GLOBAL_ERROR_LIST.append(f"[{worker_pid}] {msg}")
        return tensor_file_path

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- 4. Read Needed Frames ---
    frames_dict = {}
    needed_frame_indices = sorted([entry["frame_idx"] for entry in frame_data])
    if not needed_frame_indices:
         cap.release()
         return None # No frames specified

    max_frame_idx = needed_frame_indices[-1]
    current_frame_idx = 0
    needed_set = set(needed_frame_indices)

    while current_frame_idx <= max_frame_idx:
        ret, frame = cap.read()
        if not ret:
            # print(f"Warning [{worker_pid}]: Video ended prematurely before frame {current_frame_idx} in {video_path}")
            break # Reached end of video
        if current_frame_idx in needed_set:
            frames_dict[current_frame_idx] = frame # Keep frame in BGR
        current_frame_idx += 1
    cap.release() # Release video capture handle

    # Check if all needed frames were loaded
    if len(frames_dict) != len(needed_frame_indices):
         missing_frames = needed_set - set(frames_dict.keys())
         # print(f"Warning [{worker_pid}]: Missing frames {missing_frames} from {video_path}")
         # Decide whether to continue or fail
         # For now, we continue with available frames

    # --- 5. Prepare Crops for Batch DINO Inference ---
    dino_input_batches = []
    original_indices = [] # Keep track of which frame/entry corresponds to each crop

    tensor_idx_map = {entry['frame_idx']: i for i, entry in enumerate(frame_data)}

    # Use needed_frame_indices to ensure order matches frames_dict access
    for frm_idx in needed_frame_indices:
        if frm_idx not in frames_dict: continue # Skip if frame wasn't read

        entry = next((e for e in frame_data if e['frame_idx'] == frm_idx), None)
        if not entry: continue # Should not happen if needed_frame_indices is derived correctly

        frame = frames_dict[frm_idx]
        bbox = entry["bbox"]
        segmentation_data = entry["segmentation"]

        x1, y1, x2, y2 = expand_bbox(bbox, frame_width, frame_height, ARGS.bbox_expand_ratio)
        if x1 >= x2 or y1 >= y2: continue # Invalid bbox

        # Create mask based on segmentation (and optionally keypoints)
        # This part is kept similar to the original logic
        horse_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        try:
            points = np.asarray(segmentation_data, dtype=np.int32).reshape((-1, 2))
            cv2.fillPoly(horse_mask, [points], 255)
        except Exception as e:
            # print(f"Warning [{worker_pid}]: Error processing segmentation for frame {frm_idx}: {e}")
            continue # Skip this frame if segmentation is bad

        tensor_idx = tensor_idx_map[frm_idx]
        # Ground truth below keypoint (optional, depends on requirements)
        kp_2d_orig = pred_data['kp_2d'][tensor_idx] # KPs relative to 256x256 crop
        kp_2d_resized = resize_keypoints(kp_2d_orig, (ARGS.imgsize, ARGS.imgsize), (frame_height, frame_width))
        reference_keypoint_index = 14 # Example
        if reference_keypoint_index < len(kp_2d_resized):
            _, ref_y = kp_2d_resized[reference_keypoint_index]
            ref_y = max(0, min(ref_y, frame_height - 1))
            below_keypoint_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            below_keypoint_mask[ref_y:, :] = 255
            final_keep_mask = cv2.bitwise_or(horse_mask, below_keypoint_mask)
        else:
            final_keep_mask = horse_mask # Fallback if keypoint is missing

        # Apply mask and crop
        masked_frame = cv2.bitwise_and(frame, frame, mask=final_keep_mask)
        crop = masked_frame[y1:y2, x1:x2]
        


        if crop.size == 0: continue # Skip if crop is empty

        # Prepare for DINO: Convert BGR crop to RGB PIL Image
        try:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(crop_rgb)
            img_tensor = dino_transform(img_pil) # Apply DINO transforms
            dino_input_batches.append(img_tensor)
            original_indices.append(frm_idx) # Store the frame index
        except Exception as e:
            # print(f"Warning [{worker_pid}]: Error converting/transforming crop for frame {frm_idx}: {e}")
            continue

    if not dino_input_batches:
        print(f"Info [{worker_pid}]: No valid crops generated for DINO in {video_path}")
        # Clean up frame dict
        del frames_dict
        return None # Nothing more to process

    # --- 6. Perform Batched DINO Inference ---
    all_dino_preds = {}
    num_batches = (len(dino_input_batches) + ARGS.dino_batch_size - 1) // ARGS.dino_batch_size

    # print(dino_input_batches)



    try:
        for i in range(num_batches):
            start_idx = i * ARGS.dino_batch_size
            end_idx = min((i + 1) * ARGS.dino_batch_size, len(dino_input_batches))
            batch_tensors = torch.stack(dino_input_batches[start_idx:end_idx])
            batch_indices = original_indices[start_idx:end_idx]

            embeddings = get_dino_embedding_batch(batch_tensors, GLOBAL_MODEL)
            
            dino_results_batch = predict_orientation_dino_batch(embeddings, PROTOTYPES)

            # Store results mapped by original frame index
            for frm_idx, result in zip(batch_indices, dino_results_batch):
                all_dino_preds[frm_idx] = result

            # Clear tensors from memory if possible
            del batch_tensors, embeddings
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    except Exception as e:
         msg = f"Error during batched DINO inference for {video_path}: {e}"
         print(msg)
         GLOBAL_ERROR_LIST.append(f"[{worker_pid}] {msg}")
         del frames_dict
         del dino_input_batches
         return tensor_file_path # Indicate failure

    # Free memory used by input tensors
    del dino_input_batches


    
    output_file = video_path.replace('data', 'videos')
    output_base = output_file[:-4]
    output_dir_name = os.path.dirname(output_file)
    os.makedirs(output_dir_name, exist_ok=True)

    # --- 7. Process Frames and Write Output Videos ---
    output_writers = {}
    output_paths = {}
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    orientations = ['left', 'right', 'front', 'back', 'unknown'] # Add unknown if needed

    # Define desired output size (e.g., original size or a fixed size)
    output_height, output_width = 256, 256 # Or ARGS.imgsize, ARGS.imgsize


    try:
        # Initialize writers
        for orient in orientations:
            out_path = f"{output_base}_{orient}.mp4"
            output_paths[orient] = out_path
            output_writers[orient] = cv2.VideoWriter(out_path, fourcc, ARGS.output_fps, (output_width, output_height))

        recent_preds = collections.deque(maxlen=ARGS.smoothing_window)
        prev_moving_kp2d = None # For movement detection

        # Loop through frames again, using pre-calculated DINO results
        for frm_idx in needed_frame_indices:
            if frm_idx not in frames_dict or frm_idx not in all_dino_preds:
                continue # Skip if frame or DINO pred is missing

            frame = frames_dict[frm_idx] # Original BGR frame
            dino_pred = all_dino_preds[frm_idx]

            tensor_idx = tensor_idx_map[frm_idx]

            # Get Keypoints (ensure they exist and are valid)
            try:
                # KPs relative to 3D space or 256x256 crop? Adjust accordingly.
                # Assuming pred_kp3d_crop is available and relevant
                kp_3d = pred_data['pred_kp3d_crop'][tensor_idx] # Keep on CPU/GPU? Move as needed
                kp_2d_crop = pred_data['kp_2d'][tensor_idx] # Relative to 256x256
            except (KeyError, IndexError):
                # print(f"Warning [{worker_pid}]: Keypoints missing for frame {frm_idx}")
                kp_3d = None
                kp_2d_crop = None # Use None to indicate missing data


            # Orientation from Keypoints
            orientation_kp = "unknown"
            if kp_3d is not None:
                 orientation_kp = get_horse_orientation(kp_3d, conf=ARGS.orientation_conf) # TODO: check units of dx,dz. How was the confidence value decided?
            


            # Orientation from DINO
            orientation_dino = dino_pred['orientation']
            # score_dino = dino_pred['orientation_score'] # Available if needed

            # Movement Detection (using 2D keypoints from the 256 crop space)
            mean_displacement = np.inf # Default to moving
            if kp_2d_crop is not None:
                if prev_moving_kp2d is not None:
                     # Ensure shapes match before calculating distance
                     if kp_2d_crop.shape == prev_moving_kp2d.shape:
                         distances = np.linalg.norm(kp_2d_crop.cpu().numpy() - prev_moving_kp2d, axis=1)
                         mean_displacement = np.mean(distances) if distances.size > 0 else np.inf
                     else:
                          mean_displacement = np.inf # Reset if shape changes

                # Update previous *only if moving*
                if mean_displacement >= ARGS.movement_threshold:
                    prev_moving_kp2d = kp_2d_crop.cpu().numpy().copy() # Store CPU numpy copy
            else:
                 prev_moving_kp2d = None # Reset if KPs are missing


            # Combine Orientation Logic (Prioritize KP Left/Right if moving, else DINO)
            current_pred = "unknown"
            effective_move_thresh = ARGS.movement_threshold # Default threshold

            if orientation_kp in ["left", "right"] and mean_displacement >= effective_move_thresh:
                 current_pred = orientation_kp
            else:
                 # If not clearly left/right from KPs or not moving enough, rely on DINO
                 current_pred = orientation_dino
                 # Optionally adjust movement threshold if DINO suggests non-side view?
                 # if current_pred in ["front", "back"]: effective_move_thresh = max(1.0, ARGS.movement_threshold * 0.7)


            # Temporal Smoothing
            recent_preds.append(current_pred)
            try:
                smoothed_orientation = mode(recent_preds)
            except StatisticsError: # Handle empty or all unique values in deque
                smoothed_orientation = current_pred

            # Prepare frame for writing (Resize if output size differs from input)
            if frame.shape[0] != output_height or frame.shape[1] != output_width:
                 write_frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
            else:
                 write_frame = frame

            # Write to the corresponding video file
            writer = output_writers.get(smoothed_orientation)
            if writer:
                 writer.write(write_frame)
            else:
                 # Write to 'unknown' if orientation is not recognized (or handle as error)
                 output_writers['unknown'].write(write_frame)


    except Exception as e:
         msg = f"Error during frame processing/writing for {video_path}: {e}"
         print(msg)
         GLOBAL_ERROR_LIST.append(f"[{worker_pid}] {msg}")
         # Ensure writers are released even if an error occurs mid-loop
    finally:
        # --- 8. Release Resources ---
        for writer in output_writers.values():
            if writer: writer.release()
        del frames_dict # Free memory
        del all_dino_preds
        # print(f"[{worker_pid}] Finished processing: {tensor_file_path}")

    return None # Return None on success


class VideoDataset(Dataset):
    def __init__(self, tensor_file_list):
        self.tensor_file_list = tensor_file_list

    def __len__(self):
        return len(self.tensor_file_list)

    def __getitem__(self, idx):
        tensor_file_path = self.tensor_file_list[idx]
        try:
            # <<< --- CALL process_video HERE --- >>>
            result = process_video(tensor_file_path)
            # process_video returns None on success, path on error
            # You might want __getitem__ to return something meaningful,
            # like the path processed or the result, although often for
            # side-effect tasks (like saving files), returning None is fine.
            # If process_video raises an exception, DataLoader will catch it.
            return tensor_file_path # Return the path to indicate it was attempted
        except Exception as e:
            # Optionally handle exceptions here, maybe return an error indicator
            # Or let DataLoader handle the exception propagation
            worker_pid = os.getpid()
            error_msg = f"[{worker_pid}] Uncaught exception in __getitem__ processing {tensor_file_path}: {e}"
            print(error_msg)
            # Log error centrally if possible (though GLOBAL_ERROR_LIST might not be accessible here easily)
            # Re-raise or return an error marker
            # For simplicity, re-raising is often okay; DataLoader catches worker errors.
            raise e # Let DataLoader handle the worker failure



# Custom collate function that just passes the paths through
def collate_fn(batch):
    return batch


# --- Main Execution ---
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True) # 'spawn' is safer for CUDA context
    args = parse_args()

    # --- Setup ---
    tensor_root = os.path.join(args.data_dir, 'data_tensor')
    tensor_files = list_pt_files(tensor_root)[:10]
    
    if not tensor_files:
        print("No .pt files found to process. Exiting.")
        sys.exit(0)

    if args.limit_videos:
        tensor_files = tensor_files[:args.limit_videos]
        print(f"Processing limited to {len(tensor_files)} videos.")

    # Create a Manager process to share the error list
    manager = mp.Manager()
    shared_error_list = manager.list()

    # --- Create Dataset and DataLoader ---
    dataset = VideoDataset(tensor_files)


    init_fn = functools.partial(init_worker,
                            args_dict=vars(args),
                            error_list=shared_error_list)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size, # Usually 1 for video processing per worker call
        num_workers=args.num_workers,
        shuffle=False, # Process in defined order
        pin_memory=True, # Helps speed up CPU->GPU transfer if needed later (DINO uses it)
        prefetch_factor=2 if args.num_workers > 0 else None, # Preload items
        worker_init_fn=init_fn,
        collate_fn=collate_fn,
        persistent_workers=True if args.num_workers > 0 else False # Keep workers alive
    )



    print(f"Starting processing with {args.num_workers} workers...")
    processing_exception = None # Variable to store exception if loop fails

    # --- Run Processing ---
    try:
        for batch_paths in tqdm(dataloader, total=len(dataloader)):
            pass # Let workers handle processing
    except Exception as e:
         print(f"\n--- An error occurred during DataLoader iteration: {e} ---")
         import traceback
         traceback.print_exc()
         print("Check worker logs or error list for more details.")
         processing_exception = e # Store exception to potentially re-raise later if needed

    # --- Retrieve results BEFORE shutting down the manager ---
    final_error_list = []
    try:
        # Access the shared list while the manager is still running
        final_error_list = list(shared_error_list)
        print(f"Retrieved {len(final_error_list)} errors from shared list.")
    except Exception as e:
        print(f"\n--- Error retrieving data from shared list: {e} ---")
        print("The manager process might have crashed or become unresponsive.")
        import traceback
        traceback.print_exc()
        # final_error_list will remain empty or partially filled

    # --- Ensure Manager Shutdown (use a separate finally for this) ---
    try:
        print("Shutting down multiprocessing manager...")
        manager.shutdown()
        print("Manager shut down.")
    except Exception as e:
        print(f"\n--- Error shutting down manager: {e} ---")
        import traceback
        traceback.print_exc()


    # --- Report Errors ---
    if final_error_list:
        print("\n--- Processing Errors Occurred ---")
        # Print only unique errors if the list is long
        unique_errors = sorted(list(set(final_error_list)))
        max_errors_to_print = 50
        for i, error_msg in enumerate(unique_errors):
            print(error_msg)
            if i >= max_errors_to_print -1:
                print(f"... (truncated {len(unique_errors) - max_errors_to_print} more unique errors)")
                break

        try:
            with open(args.error_log_file, 'wb') as f:
                pickle.dump(final_error_list, f)
            print(f"\nFull error list saved to {args.error_log_file}")
        except Exception as e:
            print(f"Failed to save error log: {e}")
    elif processing_exception is None:
        # Only print success if the loop also completed without error
        print("\n--- Video processing completed with no reported errors. ---")
    else:
        # Loop had an error, but shared list might be empty (e.g., worker init failed before processing)
         print("\n--- Video processing finished with errors during execution (see above). ---")
