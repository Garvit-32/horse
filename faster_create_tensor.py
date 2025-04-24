# import argparse
# import os
# import sys
# import json
# import cv2
# import numpy as np
# import torch
# import pickle
# import multiprocessing as mp
# from torchvision import transforms
# from tqdm import tqdm

# # Adjust path to import InferenceModel
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from src.trainer.vanillaupdatetrainer import InferenceModel

# # Normalization transform
# img_mean = np.array([0.485, 0.456, 0.406])
# img_std = np.array([0.229, 0.224, 0.225])
# norm_transform = transforms.Normalize(img_mean, img_std)


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--ckpt_file', type=str, help='checkpoint for resuming')
#     parser.add_argument('--train', type=str, help='Train or Test')
#     parser.add_argument('--data_dir', type=str, default=None)
#     parser.add_argument('--model_dir', type=str,
#                         default='/home/x_cili/x_cili_lic/DESSIE/code/src/SMAL/smpl_models')
#     parser.add_argument('--save_dir', type=str, default='/home/x_cili/x_cili_lic/DESSIE/results/model')
#     parser.add_argument('--name', type=str, default='test')
#     parser.add_argument('--version', type=str)
#     parser.add_argument('--ModelName', type=str)
#     parser.add_argument('--DatasetName', type=str, default='None')
#     parser.add_argument('--batch_size', type=int, default=64)
#     parser.add_argument('--num_workers', type=int, default=4)
#     parser.add_argument('--imgsize', type=int, default=256)
#     parser.add_argument('--W_shape_prior', default=50., type=float)
#     parser.add_argument('--W_kp_img', default=0.001, type=float)
#     parser.add_argument('--W_mask_img', default=0.0001, type=float)
#     parser.add_argument('--W_pose_img', default=0.01, type=float)
#     parser.add_argument('--W_l2_shape_1', default=0., type=float)
#     parser.add_argument('--W_l2_pose_2', default=0., type=float)
#     parser.add_argument('--W_l2_shape_3', default=0., type=float)
#     parser.add_argument('--W_l2_pose_3', default=0., type=float)
#     parser.add_argument('--W_l2_rootrot_1', default=0, type=float)
#     parser.add_argument('--W_l2_rootrot_2', default=0, type=float)
#     parser.add_argument('--lr', type=float, default=5e-05)
#     parser.add_argument('--max-epochs', type=int, default=1000)
#     parser.add_argument('--seed', type=int, default=0)
#     parser.add_argument('--PosePath', type=str, default='/home/x_cili/x_cili_lic/DESSIE/data/syndata/pose')
#     parser.add_argument("--TEXTUREPath", type=str, default="/home/x_cili/x_cili_lic/DESSIE/data/syndata/TEXTure")
#     parser.add_argument('--uv_size', type=int, default=256)
#     parser.add_argument('--data_batch_size', type=int, default=2)
#     parser.add_argument('--useSynData', action="store_true")
#     parser.add_argument('--useinterval', type=int, default=8)
#     parser.add_argument("--getPairs", action="store_true", default=False)
#     parser.add_argument("--TEXT", action="store_true", default=False)
#     parser.add_argument("--DINO_model_name", type=str, default="dino_vits8")
#     parser.add_argument("--DINO_frozen", action="store_true", default=False)
#     parser.add_argument("--DINO_obtain_token", action="store_true", default=False)
#     parser.add_argument("--GT", action="store_true", default=False)
#     parser.add_argument('--W_gt_shape', default=0, type=float)
#     parser.add_argument('--W_gt_pose', default=0., type=float)
#     parser.add_argument('--W_gt_trans', default=0., type=float)
#     parser.add_argument("--pred_trans", action="store_true", default=False)
#     parser.add_argument("--background", action="store_true", default=False)
#     parser.add_argument("--background_path", default='/home/x_cili/x_cili_lic/DESSIE/data/syndata/coco')
#     parser.add_argument("--REALDATASET", default='MagicPony')
#     parser.add_argument("--REALPATH", default='/home/x_cili/x_cili_lic/DESSIE/data/realimg')
#     parser.add_argument("--web_images_num", type=int, default=0)
#     parser.add_argument("--REALMagicPonyPATH", default='/home/x_cili/x_cili_lic/DESSIE/data/magicpony')
#     return parser.parse_args()


# def set_default_args(args):
#     args.DatasetName = 'DessiePIPE'
#     args.ModelName = 'DESSIE'
#     args.name = 'TOTALRANDOM'
#     args.useSynData = True
#     args.TEXT = True
#     args.DINO_model_name = 'dino_vits8'
#     args.imgsize = 256
#     args.GT = True
#     args.pred_trans = True
#     args.W_shape_prior = 0.0
#     args.W_kp_img = 0.0
#     args.W_pose_img = 0.0
#     args.W_mask_img = 0.0
#     args.W_cos_shape = 0.0
#     args.W_cos_pose = 0.0
#     args.W_text_shape = 0.0
#     args.W_text_pose = 0.0
#     args.W_text_cam = 0.0
#     args.W_cosine_text_shape = 0.0
#     args.W_cosine_text_pose = 0.0
#     args.W_cosine_text_cam = 0.0
#     args.W_gt_shape = 0.0
#     args.W_gt_pose = 0.0
#     args.W_gt_trans = 0.0
#     args.W_l2_shape_1 = 0.0
#     args.W_l2_pose_2 = 0.0
#     args.W_l2_shape_3 = 0.0
#     args.W_l2_pose_3 = 0.0
#     args.W_l2_rootrot_1 = 0.0
#     args.W_l2_rootrot_2 = 0.0
#     args.batch_size = 1
#     args.data_batch_size = 1
#     args.getPairs = True
#     args.model_dir = '/home/ubuntu/workspace/Dessie/internal_models'
#     return args


# def convert_3d_to_2d(points, size=256):
#     focal = 5000
#     R = np.array([[-1, 0, 0],
#                   [0, -1, 0],
#                   [0, 0, 1]])
#     points_cam = points @ R.T
#     x_cam = points_cam[:, 0]
#     y_cam = points_cam[:, 1]
#     z_cam = points_cam[:, 2]
#     x_proj = (focal * x_cam) / z_cam
#     y_proj = (focal * y_cam) / z_cam
#     screen_x = (size / 2) + x_proj
#     screen_y = (size / 2) - y_proj
#     proj_points = np.stack([screen_x, screen_y], axis=-1)
#     return proj_points.astype(np.int32)


# def list_mp4_files(root_dir):
#     files = []
#     for dp, _, filenames in os.walk(root_dir):
#         for f in filenames:
#             if f.lower().endswith('.mp4'):
#                 tensor_path = os.path.join(dp, f).replace('data', 'data_tensor').replace('.mp4', '_results.pt')
#                 if not os.path.exists(tensor_path):
#                     files.append(os.path.join(dp, f))
#     return files


# def expand_bbox(bbox, frame_width, frame_height, expand_ratio=0.05):
#     x1, y1, x2, y2 = bbox
#     w_box = x2 - x1
#     h_box = y2 - y1
#     ex_w = int(w_box * expand_ratio)
#     ex_h = int(h_box * expand_ratio)
#     x1_new = max(0, x1 - ex_w)
#     y1_new = max(0, y1 - ex_h)
#     x2_new = min(frame_width, x2 + ex_w)
#     y2_new = min(frame_height, y2 + ex_h)
#     return [x1_new, y1_new, x2_new, y2_new]


# def process_batch(frames, model):
#     tensor_batch = torch.stack(frames).cuda()
#     with torch.no_grad():
#         data = model.latentpredict(tensor_batch)
#         pred_data = model.easypredict(xf_shape=data['xf_shape'],
#                                       xf_pose=data['xf_pose'],
#                                       xf_cam=data['xf_cam'],
#                                       xf=None)
#         pred_data["kp_2d"] = [convert_3d_to_2d(kp.cpu().numpy())
#                               for kp in pred_data["pred_kp3d_crop"]]
#         pred_data["xf_shape"] = data["xf_shape"]
#         pred_data["xf_pose"] = data["xf_pose"]
#         pred_data["xf_cam"] = data["xf_cam"]
#     return pred_data


# def init_worker(model_path, args_dict, error_list):
#     global GLOBAL_MODEL, GLOBAL_ERROR_LIST
#     GLOBAL_ERROR_LIST = error_list
#     args = argparse.Namespace(**args_dict)
#     args = set_default_args(args)
#     GLOBAL_MODEL = InferenceModel(args).cuda()
#     state_dict = torch.load(model_path, map_location='cpu')["state_dict"]
#     GLOBAL_MODEL.load_state_dict(state_dict)
#     GLOBAL_MODEL.eval()
#     GLOBAL_MODEL.initial_setup()


# def process_video_optimized(video_path, batch_size=16):
#     global GLOBAL_MODEL, GLOBAL_ERROR_LIST
#     json_path = video_path.replace('data', 'data_cropped').replace('.mp4', '_seg.json')
#     try:
#         with open(json_path, 'r') as f:
#             frame_data = json.load(f)
#     except Exception:
#         GLOBAL_ERROR_LIST.append(json_path)
#         print(f"Error: Cannot read JSON {json_path}")
#         return

#     needed_frames = {entry["frame_idx"] for entry in frame_data}
#     if not needed_frames:
#         print(f"Warning: No frame indices found in {json_path}")
#         return

#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         GLOBAL_ERROR_LIST.append(video_path)
#         print(f"Error: Cannot open video {video_path}")
#         return

#     frames_dict = {}
#     idx = 0
#     max_frame = max(needed_frames)
#     while idx <= max_frame:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if idx in needed_frames:
#             frames_dict[idx] = frame
#         idx += 1
#     cap.release()

#     processed_results = []
#     frames = []
#     for entry in frame_data:
#         frm_idx = entry["frame_idx"]
#         bbox = entry["bbox"]
#         frame = frames_dict.get(frm_idx)
#         if frame is None:
#             print(f"Warning: Frame {frm_idx} missing in {video_path}")
#             continue
#         h, w, _ = frame.shape
#         x1, y1, x2, y2 = expand_bbox(bbox, w, h)
#         crop = frame[y1:y2, x1:x2]
#         crop = cv2.resize(crop, (256, 256))[:, :, ::-1]  # BGR to RGB
#         crop = crop / 255.0
#         tensor = torch.from_numpy(crop).permute(2, 0, 1).float()
#         tensor = norm_transform(tensor)
#         frames.append(tensor)
#         if len(frames) >= batch_size:
#             processed_results.append(process_batch(frames, GLOBAL_MODEL))
#             frames = []
#     if frames:
#         processed_results.append(process_batch(frames, GLOBAL_MODEL))

#     if processed_results:
#         final_results = {}
#         for key in processed_results[0]:
#             vals = [res[key] for res in processed_results]
#             if isinstance(vals[0], list):
#                 vals = [torch.tensor(np.array(v)) for v in vals]
#             final_results[key] = torch.cat(vals, dim=0)
#         out_dir = os.path.dirname(video_path).replace('data', 'data_tensor')
#         os.makedirs(out_dir, exist_ok=True)
#         save_path = video_path.replace('data', 'data_tensor').replace('.mp4', '_results.pt')
#         torch.save(final_results, save_path)
#     else:
#         GLOBAL_ERROR_LIST.append(video_path)
#         print(f"Warning: No results for {video_path}")


# def process_video_wrapper(args):
#     return process_video_optimized(*args)


# if __name__ == "__main__":
#     args = parse_args()
#     args = set_default_args(args)
#     model_path = "/home/ubuntu/workspace/Dessie/COMBINAREAL/version_8/checkpoints/best.ckpt"
#     video_root = '/home/ubuntu/workspace/data'
#     video_list = list_mp4_files(video_root)
#     num_processes = 6

#     manager = mp.Manager()
#     shared_error_list = manager.list()

#     process_args = [(vp, 24) for vp in video_list]

#     with mp.Pool(processes=num_processes,
#                  initializer=init_worker,
#                  initargs=(model_path, vars(args), shared_error_list)) as pool:
#         list(tqdm(pool.imap(process_video_wrapper, process_args), total=len(process_args)))

#     print(list(shared_error_list))
#     with open('error_file_path.pkl', 'wb') as f:
#         pickle.dump(list(shared_error_list), f)

#     print("Video processing completed.")

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

# Enable cudnn benchmark for faster runtime (if input sizes are fixed)
torch.backends.cudnn.benchmark = True

# Adjust path to import InferenceModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.trainer.vanillaupdatetrainer import InferenceModel

# Normalization transform
img_mean = np.array([0.485, 0.456, 0.406])
img_std  = np.array([0.229, 0.224, 0.225])
norm_transform = transforms.Normalize(img_mean, img_std)

# Global variables to be set by worker_init_fn
GLOBAL_MODEL = None
ARGS = None
MODEL_PATH = None
GLOBAL_ERROR_LIST = []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_file', type=str)
    parser.add_argument('--train', type=str)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--model_dir', type=str,
                        default='/home/x_cili/x_cili_lic/DESSIE/code/src/SMAL/smpl_models')
    parser.add_argument('--save_dir', type=str,
                        default='/home/x_cili/x_cili_lic/DESSIE/results/model')
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--version', type=str)
    parser.add_argument('--ModelName', type=str)
    parser.add_argument('--DatasetName', type=str, default='None')
    parser.add_argument('--batch_size', type=int, default=28)  # Increased batch size
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--imgsize', type=int, default=256)
    parser.add_argument('--W_shape_prior', default=50., type=float)
    parser.add_argument('--W_kp_img', default=0.001, type=float)
    parser.add_argument('--W_mask_img', default=0.0001, type=float)
    parser.add_argument('--W_pose_img', default=0.01, type=float)
    parser.add_argument('--W_l2_shape_1', default=0., type=float)
    parser.add_argument('--W_l2_pose_2', default=0., type=float)
    parser.add_argument('--W_l2_shape_3', default=0., type=float)
    parser.add_argument('--W_l2_pose_3', default=0., type=float)
    parser.add_argument('--W_l2_rootrot_1', default=0, type=float)
    parser.add_argument('--W_l2_rootrot_2', default=0, type=float)
    parser.add_argument('--lr', type=float, default=5e-05)
    parser.add_argument('--max-epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--PosePath', type=str,
                        default='/home/x_cili/x_cili_lic/DESSIE/data/syndata/pose')
    parser.add_argument("--TEXTUREPath", type=str,
                        default="/home/x_cili/x_cili_lic/DESSIE/data/syndata/TEXTure")
    parser.add_argument('--uv_size', type=int, default=256)
    parser.add_argument('--data_batch_size', type=int, default=1)
    parser.add_argument('--useSynData', action="store_true")
    parser.add_argument('--useinterval', type=int, default=8)
    parser.add_argument("--getPairs", action="store_true", default=False)
    parser.add_argument("--TEXT", action="store_true", default=False)
    parser.add_argument("--DINO_model_name", type=str, default="dino_vits8")
    parser.add_argument("--DINO_frozen", action="store_true", default=False)
    parser.add_argument("--DINO_obtain_token", action="store_true", default=False)
    parser.add_argument("--GT", action="store_true", default=False)
    parser.add_argument('--W_gt_shape', default=0, type=float)
    parser.add_argument('--W_gt_pose', default=0., type=float)
    parser.add_argument('--W_gt_trans', default=0., type=float)
    parser.add_argument("--pred_trans", action="store_true", default=False)
    parser.add_argument("--background", action="store_true", default=False)
    parser.add_argument("--background_path", default='/home/x_cili/x_cili_lic/DESSIE/data/syndata/coco')
    parser.add_argument("--REALDATASET", default='MagicPony')
    parser.add_argument("--REALPATH", default='/home/x_cili/x_cili_lic/DESSIE/data/realimg')
    parser.add_argument("--web_images_num", type=int, default=0)
    parser.add_argument("--REALMagicPonyPATH", default='/home/x_cili/x_cili_lic/DESSIE/data/magicpony')
    return parser.parse_args()


def set_default_args(args):
    args.DatasetName = 'DessiePIPE'
    args.ModelName = 'DESSIE'
    args.name = 'TOTALRANDOM'
    args.useSynData = True
    args.TEXT = True
    args.DINO_model_name = 'dino_vits8'
    args.imgsize = 256
    args.GT = True
    args.pred_trans = True
    args.W_shape_prior = 0.0
    args.W_kp_img = 0.0
    args.W_pose_img = 0.0
    args.W_mask_img = 0.0
    args.W_cos_shape = 0.0
    args.W_cos_pose = 0.0
    args.W_text_shape = 0.0
    args.W_text_pose = 0.0
    args.W_text_cam = 0.0
    args.W_cosine_text_shape = 0.0
    args.W_cosine_text_pose = 0.0
    args.W_cosine_text_cam = 0.0
    args.W_gt_shape = 0.0
    args.W_gt_pose = 0.0
    args.W_gt_trans = 0.0
    args.W_l2_shape_1 = 0.0
    args.W_l2_pose_2 = 0.0
    args.W_l2_shape_3 = 0.0
    args.W_l2_pose_3 = 0.0
    args.W_l2_rootrot_1 = 0.0
    args.W_l2_rootrot_2 = 0.0
    args.batch_size = args.batch_size  # Use provided batch size
    args.data_batch_size = 1
    args.getPairs = True
    args.model_dir = '/home/ubuntu/workspace/Dessie/internal_models'
    return args


def convert_3d_to_2d(points, size=256):
    focal = 5000
    R = np.array([[-1, 0, 0],
                  [0, -1, 0],
                  [0, 0, 1]])
    points_cam = points @ R.T
    x_cam = points_cam[:, 0]
    y_cam = points_cam[:, 1]
    z_cam = points_cam[:, 2]
    x_proj = (focal * x_cam) / z_cam
    y_proj = (focal * y_cam) / z_cam
    screen_x = (size / 2) + x_proj
    screen_y = (size / 2) - y_proj
    proj_points = np.stack([screen_x, screen_y], axis=-1)
    return proj_points.astype(np.int32)


def list_mp4_files(root_dir):
    files = []
    for dp, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith('.mp4'):
                tensor_path = os.path.join(dp, f).replace('data', 'data_tensor').replace('.mp4', '_results.pt')
                if not os.path.exists(tensor_path):
                    files.append(os.path.join(dp, f))
    return files


def expand_bbox(bbox, frame_width, frame_height, expand_ratio=0.05):
    x1, y1, x2, y2 = bbox
    w_box = x2 - x1
    h_box = y2 - y1
    ex_w = int(w_box * expand_ratio)
    ex_h = int(h_box * expand_ratio)
    x1_new = max(0, x1 - ex_w)
    y1_new = max(0, y1 - ex_h)
    x2_new = min(frame_width, x2 + ex_w)
    y2_new = min(frame_height, y2 + ex_h)
    return [x1_new, y1_new, x2_new, y2_new]


def process_batch(frames, model):
    device = torch.device("cuda")
    # Use non_blocking transfer since DataLoader uses pinned memory.
    tensor_batch = torch.stack(frames).to(device, non_blocking=True)
    # Use automatic mixed precision for faster inference.
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            data = model.latentpredict(tensor_batch)
            pred_data = model.easypredict(xf_shape=data['xf_shape'],
                                          xf_pose=data['xf_pose'],
                                          xf_cam=data['xf_cam'],
                                          xf=None)
            pred_data["kp_2d"] = [convert_3d_to_2d(kp.cpu().numpy()) for kp in pred_data["pred_kp3d_crop"]]
            pred_data["xf_shape"] = data["xf_shape"]
            pred_data["xf_pose"] = data["xf_pose"]
            pred_data["xf_cam"] = data["xf_cam"]
    return pred_data


def init_worker(model_path, args_dict):
    global GLOBAL_MODEL, ARGS, MODEL_PATH, GLOBAL_ERROR_LIST
    GLOBAL_ERROR_LIST = []  # reset error list per worker if needed
    ARGS = argparse.Namespace(**args_dict)
    ARGS = set_default_args(ARGS)
    MODEL_PATH = model_path
    GLOBAL_MODEL = InferenceModel(ARGS).cuda()
    state_dict = torch.load(MODEL_PATH, map_location='cpu')["state_dict"]
    GLOBAL_MODEL.load_state_dict(state_dict)
    GLOBAL_MODEL.eval()
    GLOBAL_MODEL.initial_setup()


def process_video(video_path, batch_size):
    global GLOBAL_MODEL, GLOBAL_ERROR_LIST
    json_path = video_path.replace('data', 'data_cropped').replace('.mp4', '_seg.json')
    try:
        with open(json_path, 'r') as f:
            frame_data = json.load(f)
    except Exception:
        GLOBAL_ERROR_LIST.append(json_path)
        print(f"Error: Cannot read JSON {json_path}")
        return video_path

    needed_frames = {entry["frame_idx"] for entry in frame_data}
    if not needed_frames:
        print(f"Warning: No frame indices in {json_path}")
        return video_path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        GLOBAL_ERROR_LIST.append(video_path)
        print(f"Error: Cannot open video {video_path}")
        return video_path

    frames_dict = {}
    idx = 0
    max_frame = max(needed_frames)
    while idx <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in needed_frames:
            frames_dict[idx] = frame
        idx += 1
    cap.release()

    processed_results = []
    frames = []
    for entry in frame_data:
        frm_idx = entry["frame_idx"]
        bbox = entry["bbox"]
        frame = frames_dict.get(frm_idx)
        if frame is None:
            print(f"Warning: Frame {frm_idx} missing in {video_path}")
            continue
        h, w, _ = frame.shape
        x1, y1, x2, y2 = expand_bbox(bbox, w, h)
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (256, 256))[:, :, ::-1]  # Convert BGR to RGB
        crop = crop / 255.0
        tensor = torch.from_numpy(crop).permute(2, 0, 1).float()
        tensor = norm_transform(tensor)
        frames.append(tensor)
        if len(frames) >= batch_size:
            processed_results.append(process_batch(frames, GLOBAL_MODEL))
            frames = []
    if frames:
        processed_results.append(process_batch(frames, GLOBAL_MODEL))

    if processed_results:
        final_results = {}
        for key in processed_results[0]:
            vals = [res[key] for res in processed_results]
            if isinstance(vals[0], list):
                vals = [torch.tensor(np.array(v)) for v in vals]
            final_results[key] = torch.cat(vals, dim=0)
        out_dir = os.path.dirname(video_path).replace('data', 'data_tensor')
        os.makedirs(out_dir, exist_ok=True)
        save_path = video_path.replace('data', 'data_tensor').replace('.mp4', '_results.pt')
        torch.save(final_results, save_path)
    else:
        GLOBAL_ERROR_LIST.append(video_path)
        print(f"Warning: No results for {video_path}")

    return video_path


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_list, batch_size):
        self.video_list = video_list
        self.batch_size = batch_size

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_path = self.video_list[idx]
        process_video(video_path, self.batch_size)
        return video_path


if __name__ == "__main__":
    args = parse_args()
    args = set_default_args(args)
    MODEL_PATH = "/home/ubuntu/workspace/Dessie/COMBINAREAL/version_8/checkpoints/best.ckpt"
    video_root = '/home/ubuntu/workspace/data'
    video_list = list_mp4_files(video_root)

    dataset = VideoDataset(video_list, batch_size=args.batch_size)
    # Use DataLoader with pin_memory to speed up CPU-to-GPU transfers.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        worker_init_fn=lambda worker_id: init_worker(MODEL_PATH, vars(args))
    )

    for _ in tqdm(dataloader, total=len(dataloader)):
        pass

    print("Errors:", GLOBAL_ERROR_LIST)
    with open('error_file_path.pkl', 'wb') as f:
        pickle.dump(GLOBAL_ERROR_LIST, f)

    print("Video processing completed.")
