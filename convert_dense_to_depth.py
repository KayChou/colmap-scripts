import argparse
import numpy as np
import os
import struct
from PIL import Image
import warnings
import os
import config

warnings.filterwarnings('ignore') # 屏蔽nan与min_depth比较时产生的警告

fB = 32504

camera_num = config.camera_num
frame_num = config.frame_num
path = config.path

min_depth_percentile = 2
max_depth_percentile = 98

def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def compute_min_max_depth(dense_dir):
    min_depth = 1000
    max_depth = 0
    for i in range(camera_num):
        depth_map = dense_dir + "stereo/depth_maps/" + str(i) + '.png.' + 'geometric' + '.bin'
        depth_map = read_array(depth_map)
        min_tmp, max_tmp = np.percentile(depth_map[depth_map > 0], [min_depth_percentile, max_depth_percentile])
        if(min_tmp < min_depth):
            min_depth = min_tmp
        if(max_tmp > max_depth):
            max_depth = max_tmp
    print("depth range: min: %f max %f\n"%(min_depth, max_depth))
    return min_depth, max_depth


def bin2depth(i, min_depth, max_depth, depth_map, depth_dir):
    if min_depth_percentile > max_depth_percentile:
        raise ValueError("min_depth_percentile should be less than or equal "
                         "to the max_depth_perceintile.")
    
    if not os.path.exists(depth_map):
        print("file {} not found\n".format(depth_map))
        return

    depth_map = read_array(depth_map)

    depth_map[depth_map <= 0] = np.nan # 把0和负数都设置为nan，防止被min_depth取代
    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth

    maxdisp = fB / min_depth
    mindisp = fB / max_depth
    depth_map = (fB/depth_map - mindisp) * 255 / (maxdisp - mindisp)
    depth_map = np.nan_to_num(depth_map) # nan全都变为0
    depth_map = depth_map.astype(int)

    image = Image.fromarray(np.uint8(depth_map)).convert('L')
    image.save(depth_dir + str(i) + '.png')


if __name__ == "__main__":
    min_depth, max_depth = compute_min_max_depth(path + "/frames/frame_0/dense/")
    for frame_idx in range(frame_num):
        dense_dir = path + "/frames/frame_{}/dense/".format(frame_idx)
        depth_dir = path + "/frames/frame_{}/depths/".format(frame_idx)
        triangulate_dir = path + "/frames/frame_{}/triangulated/".format(frame_idx)

        if not os.path.exists(dense_dir + "stereo/depth_maps/"):
            # print(f"frame_{frame_idx} not valid")
            print(frame_idx, ",")
            continue

        if not os.path.exists(depth_dir):
            os.mkdir(depth_dir)

        for i in range(camera_num):
            depth_path = dense_dir + "stereo/depth_maps/" + str(i) + '.png.' + 'geometric' + '.bin'
            
            bin2depth(i, min_depth, max_depth, depth_path, depth_dir) # photometric
