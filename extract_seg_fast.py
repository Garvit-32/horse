import cv2
from ultralytics import YOLO
import numpy as np
import os
from tqdm import tqdm
import multiprocessing as mp
import torch
import json

def process_video_segmentation_batch_efficient(video_path, batch_size=32, output_dir='data_cropped', model_name='yolo11x-seg.pt', conf_threshold=0.5, iou_threshold=0.5, class_id_to_detect=17, frame_skip_rate=3):
    """
    Processes a single video for segmentation efficiently using batched inference and optimized resource utilization.

    Args:
        video_path (str): Path to the input video file.
        batch_size (int): Batch size for YOLO inference. Adjust based on GPU memory.
        output_dir (str): Directory to save cropped video.
        model_name (str): Name of the YOLO segmentation model file.
        conf_threshold (float): Confidence threshold for detections.
        iou_threshold (float): IoU threshold for NMS.
        class_id_to_detect (int): Class ID to detect and crop (default: 17 for horse).
        frame_skip_rate (int): Process every nth frame (e.g., 3 for every 3rd frame).
    """

    # Load YOLOv8 segmentation model (Load model outside the loop for efficiency)
    model = YOLO(model_name, verbose=False)
    # Set device to GPU if available, otherwise CPU (explicitly set device for control)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.conf = conf_threshold  # Set confidence threshold
    model.iou = iou_threshold    # Set IoU threshold


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    dir_name = os.path.dirname(video_path).replace('data', output_dir)
    os.makedirs(dir_name, exist_ok=True)
    output_video_path = video_path.replace('data', output_dir)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    max_bbox_area = 0.
    max_bbox_height = 0.
    max_bbox_width = 0.
    all_frames_results = []
    frames_batches = []
    frames_idxs = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip_rate == 0:
            frames_batches.append(frame)
            frames_idxs.append(frame_idx)

        frame_idx += 1

    num_frames = len(frames_batches)
    if not frames_batches:  # Handle case where no frames are processed (e.g., very short video with high skip rate)
        cap.release()
        print(f"No frames processed for {video_path} (check frame_skip_rate).")
        return


    for offset in range(0, num_frames, batch_size):
        batch_indices = frames_idxs[offset:offset + batch_size]
        frames_batch = frames_batches[offset:offset + batch_size]

        results = model(frames_batch, verbose=False, stream=False, imgsz=640)  # Perform batched segmentation inference, stream=False for batched input


        for frame_idx_in_batch, result in enumerate(results):
            frame = frames_batch[frame_idx_in_batch]
            frame_idx = batch_indices[frame_idx_in_batch]


            boxes = result.boxes  # Bounding boxes
            masks = result.masks  # Segmentation masks
            polygons = masks.xy if masks is not None else []  # Segmentation masks in XY format
            scores = boxes.conf.tolist()  # Confidence scores
            class_ids = np.array(boxes.cls.tolist()).astype(int)  # Class IDs


            if np.any(class_ids == class_id_to_detect): # Use np.any for efficiency
                for i, class_id in enumerate(class_ids):
                    if int(class_id) == class_id_to_detect:
                        horse_data = {
                            "frame_idx": frame_idx,
                            # "path": video_path,
                            "confidence": scores[i],
                            "bbox": list(map(int, boxes.xyxy[i].tolist())),  # Bounding box in [x1, y1, x2, y2]
                            "segmentation": polygons[i].tolist() if i < len(polygons) else []  # Segmentation polygon, convert numpy array to list
                        }
                        all_frames_results.append(horse_data)
                        current_bbox = boxes.xyxy[i].tolist()
                        current_width = current_bbox[2] - current_bbox[0]
                        current_height = current_bbox[3] - current_bbox[1]

                        if current_height * current_width > max_bbox_area:
                            max_bbox_width = current_width
                            max_bbox_height = current_height
    ## write all frames results to json
    json_path = output_video_path.replace('.mp4', '.json')
    with open(json_path, 'w') as f:
        json.dump(all_frames_results, f)


def process_video_wrapper(video_path, batch_size, output_dir, model_name, conf_threshold, iou_threshold, class_id_to_detect, frame_skip_rate):
    """Wrapper function for process_video_segmentation_batch_efficient to handle process-based execution."""
    process_video_segmentation_batch_efficient(video_path, batch_size, output_dir, model_name, conf_threshold, iou_threshold, class_id_to_detect, frame_skip_rate)

def process_video_wrapper_star(args):
    return process_video_wrapper(*args)

def list_mp4_files(root_dir):
    mp4_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.mp4') and "_vis.mp4" not in filename and 'DLC_Resnet50_yearlingwalkJun13shuffle1_snapshot_200_p75_labeled' not in filename:
                file_path = os.path.join(dirpath, filename)
                mp4_files.append(file_path)
    return mp4_files


if __name__ == "__main__":
    video_path = '/home/ubuntu/workspace/data'
    video_list = list_mp4_files(video_path)

    batch_size = 16  # Adjust batch size based on your GPU memory
    output_dir = "data_cropped"
    model_name = 'yolo11x-seg.pt' # Consider smaller models like 'yolov8s-seg.pt' or 'yolov8m-seg.pt' for faster inference if acceptable
    conf_threshold = 0.5 # Adjust confidence threshold as needed
    iou_threshold = 0.5  # Adjust IoU threshold as needed
    class_id_to_detect = 17 # Class ID for horse
    frame_skip_rate = 3 # Process every 3rd frame


    num_processes = 16 # Use all available CPU cores
    # num_processes = 4 # Or limit the number of processes if needed for memory or other reasons

    print("number of videos", len(video_list))

    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_video_wrapper_star, 
                        [(video_file, batch_size, output_dir, model_name, conf_threshold, iou_threshold, class_id_to_detect, frame_skip_rate) 
                        for video_file in video_list]), 
            total=len(video_list)))
    print("Video processing complete.")