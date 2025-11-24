import h5py, pickle
import numpy as np
import os
import cv2
from collections.abc import Mapping, Sequence
import shutil
from .images_to_video import images_to_video
from pprint import pprint


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def parse_dict_structure(data):
    if isinstance(data, dict):
        parsed = {}
        for key, value in data.items():
            if isinstance(value, dict):
                parsed[key] = parse_dict_structure(value)
            elif isinstance(value, np.ndarray):
                parsed[key] = []
            else:
                parsed[key] = []
        return parsed
    else:
        return []


def append_data_to_structure(data_structure, data):
    for key in data_structure:
        if key in data:
            if isinstance(data_structure[key], list):
                # 如果是叶子节点，直接追加数据
                data_structure[key].append(data[key])
            elif isinstance(data_structure[key], dict):
                # 如果是嵌套字典，递归处理
                append_data_to_structure(data_structure[key], data[key])


def load_pkl_file(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def create_hdf5_from_dict(hdf5_group, data_dict):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            subgroup = hdf5_group.create_group(key)
            create_hdf5_from_dict(subgroup, value)
        elif isinstance(value, list):
            value = np.array(value)
            if "rgb" in key:
                encode_data, max_len = images_encoding(value)
                hdf5_group.create_dataset(key, data=encode_data, dtype=f"S{max_len}")
            else:
                hdf5_group.create_dataset(key, data=value)
        else:
            return
            try:
                hdf5_group.create_dataset(key, data=str(value))
                print("Not np array")
            except Exception as e:
                print(f"Error storing value for key '{key}': {e}")


def pkl_files_to_hdf5_and_video(pkl_files, hdf5_path, video_path):
    data_list = parse_dict_structure(load_pkl_file(pkl_files[0]))
    for pkl_file_path in pkl_files:
        pkl_file = load_pkl_file(pkl_file_path)
        append_data_to_structure(data_list, pkl_file)
    
    third_view_rgb = data_list.get("third_view_rgb", None)
    endpose = data_list.get("endpose", {}).get("left_endpose", None)
    if endpose is not None:
        endpose = np.asarray(endpose)  # shape (N,7)

    # unified overlay function for both single-view and stitched frames
    def overlay_endpose(frames, endpose_arr):
        if endpose_arr is None:
            return np.array(frames)
        frames = np.array(frames)
        if frames.size == 0:
            return frames
        n_frames = frames.shape[0]
        n_pose = endpose_arr.shape[0]
        out = []
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        padding = 6
        for i in range(n_frames):
            img = frames[i].copy()
            # normalize/convert dtype to uint8
            try:
                img_max = img.max()
            except Exception:
                img_max = None
            if img.dtype != np.uint8:
                if img_max is not None and img_max <= 1.0:
                    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            pose = endpose_arr[min(i, n_pose - 1)]
            lines = [f"{v:.3f}" for v in pose]
            # compute box size
            line_sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
            line_widths = [s[0] for s in line_sizes]
            line_heights = [s[1] for s in line_sizes]
            box_w = max(line_widths) + padding * 2
            box_h = sum(line_heights) + (len(lines)-1)*4 + padding * 2
            h, w = img.shape[:2]
            x0 = max(5, w - box_w - 5)
            y0 = 5
            cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
            y = y0 + padding + line_heights[0]
            for idx, line in enumerate(lines):
                cv2.putText(img, line, (x0 + padding, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
                if idx + 1 < len(lines):
                    y += line_heights[idx+1] + 4
            out.append(img)
        return np.array(out)

    if third_view_rgb is not None:
        # 与 112-160 行保存方式一致：在相同目录、相同扩展名下生成带后缀的文件名，并确保目录存在
        base, ext = os.path.splitext(video_path)
        if not ext:
            ext = ".mp4"
        third_video_path = f"{base}_third_view{ext}"
        out_dir = os.path.dirname(third_video_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        third_frames = np.array(third_view_rgb)
        third_frames = overlay_endpose(third_frames, endpose)
        images_to_video(third_frames, out_path=third_video_path)

    # Prepare stitched video from multiple camera views (left, right, head).
    obs = data_list.get("observation", {})
    cam_names = ["head_camera", "left_camera", "right_camera"]
    # Collect rgb lists for available cameras
    cam_frames = {}
    for name in cam_names:
        cam = obs.get(name, {})
        rgb_list = cam.get("rgb") if isinstance(cam, dict) else None
        if rgb_list:
            cam_frames[name] = np.array(rgb_list)

    if not cam_frames:
        # Fallback to previous behavior: try left camera path and let errors surface if missing
        left_frames = np.array(data_list["observation"]["left_camera"]["rgb"])
        left_frames = overlay_endpose(left_frames, endpose)
        images_to_video(left_frames, out_path=video_path)
    else:
        # Ensure all collected cameras have the same number of frames and same HWC shapes
        # Find minimal common length to avoid index errors
        lengths = [v.shape[0] for v in cam_frames.values()]
        n_frames = min(lengths)

        stitched = []
        for i in range(n_frames):
            frames = []
            for name in cam_names:
                if name in cam_frames:
                    frm = cam_frames[name][i]
                    # If single-channel or grayscale, convert to 3-channel
                    if frm.ndim == 2:
                        frm = np.stack([frm] * 3, axis=-1)
                    frames.append(frm)
            # Concatenate horizontally (axis=1 -> width)
            try:
                stitched_frame = np.concatenate(frames, axis=1)
            except Exception:
                # If concat fails due to mismatched heights, resize to smallest height
                heights = [f.shape[0] for f in frames]
                min_h = min(heights)
                resized = []
                for f in frames:
                    if f.shape[0] != min_h:
                        resized_f = cv2.resize(f, (int(f.shape[1] * (min_h / f.shape[0])), min_h))
                    else:
                        resized_f = f
                    resized.append(resized_f)
                stitched_frame = np.concatenate(resized, axis=1)
            stitched.append(stitched_frame)

        stitched_arr = np.array(stitched)
        stitched_arr = overlay_endpose(stitched_arr, endpose)
        images_to_video(stitched_arr, out_path=video_path)

    with h5py.File(hdf5_path, "w") as f:
        create_hdf5_from_dict(f, data_list)


def process_folder_to_hdf5_video(folder_path, hdf5_path, video_path):
    pkl_files = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".pkl") and fname[:-4].isdigit():
            pkl_files.append((int(fname[:-4]), os.path.join(folder_path, fname)))

    if not pkl_files:
        raise FileNotFoundError(f"No valid .pkl files found in {folder_path}")

    pkl_files.sort()
    pkl_files = [f[1] for f in pkl_files]

    expected = 0
    for f in pkl_files:
        num = int(os.path.basename(f)[:-4])
        if num != expected:
            raise ValueError(f"Missing file {expected}.pkl")
        expected += 1

    pkl_files_to_hdf5_and_video(pkl_files, hdf5_path, video_path)
