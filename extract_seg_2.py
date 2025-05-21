import cv2
from ultralytics import YOLO
import numpy as np
import os
from tqdm import tqdm
import multiprocessing as mp
import torch
import json

# Global variable for the model in each worker process.
GLOBAL_MODEL = None

def init_worker(model_name, conf_threshold, iou_threshold):
    """
    Initializer function for each pool worker. Loads the YOLO model once per process.
    """
    global GLOBAL_MODEL
    GLOBAL_MODEL = YOLO(model_name, verbose=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GLOBAL_MODEL.to(device)
    GLOBAL_MODEL.conf = conf_threshold
    GLOBAL_MODEL.iou = iou_threshold
    GLOBAL_MODEL.eval() 

def process_video_segmentation_batch_efficient(video_path, batch_size=32, output_dir='data_cropped',
                                               class_id_to_detect=17, frame_skip_rate=3):
    """
    Processes a single video for segmentation using batched inference.
    
    Args:
        video_path (str): Path to the input video file.
        batch_size (int): Batch size for inference.
        output_dir (str): Directory to save JSON output.
        class_id_to_detect (int): Class ID to detect.
        frame_skip_rate (int): Process every nth frame.
    """
    global GLOBAL_MODEL
    if GLOBAL_MODEL is None:
        raise RuntimeError("Model is not initialized in worker.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    # Construct a robust output path preserving the subdirectory structure.
    dir_name = os.path.dirname(video_path).replace('data', output_dir)
    os.makedirs(dir_name, exist_ok=True)
    output_video_path = video_path.replace('data', output_dir)

    all_frames_results = []
    frames_batches = []
    frames_idxs = []
    frame_idx = 0

    # Read frames while skipping as specified.
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip_rate == 0:
            frames_batches.append(frame)
            frames_idxs.append(frame_idx)
        frame_idx += 1
    cap.release()

    if not frames_batches:
        print(f"No frames processed for {video_path} (check frame_skip_rate).")
        return

    # Track maximum bounding box dimensions.
    max_bbox_area = 0
    max_bbox_dims = (0, 0)

    # Process frames in batches.
    for offset in range(0, len(frames_batches), batch_size):
        batch_indices = frames_idxs[offset:offset + batch_size]
        frames_batch = frames_batches[offset:offset + batch_size]

        # Run batched segmentation inference.
        with torch.no_grad():
            results = GLOBAL_MODEL(frames_batch, verbose=False, stream=False, imgsz=640)

        for idx, result in enumerate(results):
            current_frame_idx = batch_indices[idx]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            scores = boxes.conf.tolist()
            bboxes = boxes.xyxy.tolist()
            class_ids = np.array(boxes.cls.tolist(), dtype=int)

            # Get indices for detections of the target class.
            indices = np.where(class_ids == class_id_to_detect)[0]
            if indices.size == 0:
                continue

            # Retrieve segmentation polygons if available.
            polygons = result.masks.xy if result.masks is not None else []

            for i in indices:
                bbox = list(map(int, bboxes[i]))
                segmentation = polygons[i].tolist() if i < len(polygons) else []
                score = scores[i]

                all_frames_results.append({
                    "frame_idx": current_frame_idx,
                    "confidence": score,
                    "bbox": bbox,
                    "segmentation": segmentation
                })

                current_width = bbox[2] - bbox[0]
                current_height = bbox[3] - bbox[1]
                area = current_width * current_height
                if area > max_bbox_area:
                    max_bbox_area = area
                    max_bbox_dims = (current_width, current_height)
        torch.cuda.empty_cache()
    # Write detection results to a JSON file.
    json_path = output_video_path.replace('.mp4', '_seg.json')
    with open(json_path, 'w') as f:
        json.dump(all_frames_results, f)

    # print(f"Processed {video_path}: max bbox dims {max_bbox_dims}.")

def process_video_wrapper(args):
    return process_video_segmentation_batch_efficient(*args)

def list_mp4_files(root_dir):
    """
    Recursively lists all .mp4 files in a directory, excluding files that match specific patterns.
    """
    mp4_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if (filename.lower().endswith('.mp4') and "_vis.mp4" not in filename and
                    'DLC_Resnet50_yearlingwalkJun13shuffle1_snapshot_200_p75_labeled' not in filename):
                mp4_files.append(os.path.join(dirpath, filename))
    return mp4_files

if __name__ == "__main__":
    video_root = '/home/ubuntu/workspace/data'
    video_list = list_mp4_files(video_root)
    print("Number of videos:", len(video_list))
    video_list_copy = video_list.copy()
    for video_file in video_list:
        if os.path.exists(video_file.replace('data', 'data_cropped').replace('.mp4', '_seg.json')):
            video_list_copy.remove(video_file)
    video
    print("Number of videos to process:", len(video_list_copy))
    # # Parameters
    batch_size = 16
    output_dir = "data_cropped"
    model_name = 'yolo11x-seg.pt'
    conf_threshold = 0.5
    iou_threshold = 0.5
    class_id_to_detect = 17
    frame_skip_rate = 3

    num_processes = 16
    args_list = [
        (video_file, batch_size, output_dir, class_id_to_detect, frame_skip_rate)
        for video_file in video_list
    ]

    # Create a pool with an initializer that loads the model once per worker.
    with mp.Pool(processes=num_processes,
                 initializer=init_worker,
                 initargs=(model_name, conf_threshold, iou_threshold)) as pool:
        list(tqdm(pool.imap(process_video_wrapper, args_list), total=len(args_list)))

    print("Video processing complete.")
