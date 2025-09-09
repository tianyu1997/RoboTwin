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

    # Prepare stitched video from multiple camera views (left, right, head).
    obs = data_list.get("observation", {})
    pprint(obs)
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
        images_to_video(np.array(data_list["observation"]["left_camera"]["rgb"]), out_path=video_path)
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
                widths = [f.shape[1] for f in frames]
                min_h = min(heights)
                # resize each frame to min_h using simple crop or cv2.resize
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
