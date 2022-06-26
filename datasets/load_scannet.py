import os
import imageio
import numpy as np
import re
import cv2
import tqdm


def load_poses(posefile):
    
    file = open(posefile, "r")
    lines = file.readlines()
    file.close()
    poses = []
    valid = []
    lines_per_matrix = 4
    for i in range(0, len(lines), lines_per_matrix):
        if 'nan' in lines[i]:
            valid.append(False)
            poses.append(np.eye(4, 4, dtype=np.float32).tolist())
        else:
            valid.append(True)
            pose_floats = [[float(x) for x in line.split()] for line in lines[i:i+lines_per_matrix]]
            poses.append(pose_floats)

    return poses, valid


def load_focal_length(filepath):
    file = open(filepath, "r")
    return float(file.readline())


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [int(x) if x.isdigit() else x for x in re.split('([0-9]+)', s)]


def resize_images(images, H, W, interpolation=cv2.INTER_LINEAR):
    resized = np.zeros((images.shape[0], H, W, images.shape[3]), dtype=images.dtype)
    for i, img in enumerate(images):
        r = cv2.resize(img, (W, H), interpolation=interpolation)
        if images.shape[3] == 1:
            r = r[..., np.newaxis]
        resized[i] = r
    return resized



def get_training_poses(basedir, translation=0.0, sc_factor=1.0, trainskip=1):
    all_poses, valid = load_poses(os.path.join(basedir, 'trainval_poses.txt'))

    train_frames = []
    for idx in range(0, len(all_poses), trainskip):
        if valid[idx]:
            train_frames.append(idx)

    all_poses = np.array(all_poses).astype(np.float32)
    training_poses = all_poses[train_frames]

    training_poses[:, :3, 3] += translation
    training_poses[:, :3, 3] *= sc_factor

    return training_poses


def get_num_training_frames(basedir, trainskip):
    poses = get_training_poses(basedir, trainskip=trainskip)

    return poses.shape[0]


def get_intrinsics(basedir, crop):
    depth = imageio.imread(os.path.join(basedir, 'depth_filtered', 'depth0.png'))
    H, W = depth.shape[:2]
    H = H - crop / 2
    W = W - crop / 2
    focal = load_focal_length(os.path.join(basedir, 'focal.txt'))

    return H, W, focal


def load_scannet_data(basedir, trainskip, downsample_factor=1, translation=0.0, sc_factor=1., crop=0):

    # Get image filenames, poses and intrinsics
    img_files = [f for f in sorted(os.listdir(os.path.join(basedir, 'images')), key=alphanum_key) if f.endswith('png')]
    depth_files = [f for f in sorted(os.listdir(os.path.join(basedir, 'depth_filtered')), key=alphanum_key) if f.endswith('png')]

    # If Pose is NaN, then valid=false, initialise as 4x4 identity matrix
    all_poses, valid_poses = load_poses(os.path.join(basedir, 'trainval_poses.txt'))

    # Train, val and test split
    num_frames = len(img_files)
    train_frame_ids = list(range(0, num_frames, trainskip))

    # Lists for the data to load into
    images = []
    depth_maps = []
    poses = []
    frame_indices = []

    # Read images and depth maps for which valid poses exist
    for i in tqdm.tqdm(train_frame_ids):
        if valid_poses[i]:
            img = imageio.imread(os.path.join(basedir, 'images', img_files[i]))
            depth = imageio.imread(os.path.join(basedir, 'depth_filtered', depth_files[i]))

            images.append(img)
            depth_maps.append(depth)
            poses.append(all_poses[i])
            frame_indices.append(i)

    # Map images to [0, 1] range
    images = (np.array(images) / 255.).astype(np.float32)

    # Convert depth to meters, then to "network units"
    depth_shift = 1000.0
    depth_maps = (np.array(depth_maps) / depth_shift).astype(np.float32)
    depth_maps *= sc_factor
    depth_maps = depth_maps[..., np.newaxis]

    poses = np.array(poses).astype(np.float32)
    poses[:, :3, 3] += translation
    poses[:, :3, 3] *= sc_factor

    # Intrinsics
    H, W = depth_maps[0].shape[:2]
    focal = load_focal_length(os.path.join(basedir, 'focal.txt'))

    # Resize color frames to match depth
    images = resize_images(images, H, W)

    # Crop the undistortion artifacts
    if crop > 0:
        images = images[:, crop:-crop, crop:-crop, :]
        depth_maps = depth_maps[:, crop:-crop, crop:-crop, :]
        H, W = depth_maps[0].shape[:2]

    if downsample_factor > 1:
        H = H//downsample_factor
        W = W//downsample_factor
        focal = focal/downsample_factor
        images = resize_images(images, H, W)
        depth_maps = resize_images(depth_maps, H, W, interpolation=cv2.INTER_NEAREST)

    return images, depth_maps, poses, [H, W, focal], frame_indices
