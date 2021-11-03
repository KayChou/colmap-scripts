import os
import config
from scipy.spatial.transform import Rotation as R
import numpy as np

camera_num = config.camera_num
frame_num = config.frame_num


def convert_raw_to_RT(path, frame_idx):
    filename = path + "images.txt"

    # parse images.txt as dict: {0.png: ..., 1.png: ..., ... 11.png: ...}
    f = open(filename, 'r')
    dict = {}
    for line in f.readlines():
        if(line[-5:-1] == ".png"):
            line_list = line.strip().split(" ")
            dict[line_list[-1]] = list(map(float, line_list[1:-2]))

    
    if(len(dict) < camera_num):
        print("Only {} cameras' params are estimated, skip\n".format(len(dict)))
        return

    f.close()

    f = open("params_k_means_RT.txt", "a")

    cnt = 0
    # for cam_idx in [0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9]:\
    for cam_idx in range(camera_num):
        key = "{0:02d}.png".format(cam_idx)
        R_quat = dict[key][0:4] # R: Quaternion
        T = dict[key][4:7] # T: world position
        # euler_xyz = R.from_quat(R_quat).as_euler('xyz', degrees=True)
        R_matrix = R.from_quat(R_quat).as_matrix()
        
        f.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}.png\n".format(cnt + 1, R_matrix[0][0], R_matrix[0][1], R_matrix[0][2], R_matrix[1][0], R_matrix[1][1], R_matrix[1][2], R_matrix[2][0], R_matrix[2][1], R_matrix[2][2], T[0], T[1], T[2], cnt + 1, cam_idx))
        # f.write("{} {} {} {} {} {}.png\n".format(cnt + 1, T[0], T[1], T[2], cnt + 1, cam_idx))
        cnt = cnt + 1
    f.write("\n")
    f.close()

if __name__ == "__main__":
    # for frame_idx in range(4):
    #     sparse_dir = config.path + "/frames/frame_{}/dense/sparse/".format(frame_idx)
    #     convert_raw_to_RT(sparse_dir, frame_idx)
    convert_raw_to_RT(config.path + "/created/", 0)