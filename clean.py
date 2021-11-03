import os
import config

camera_num = config.camera_num
frame_num = config.frame_num
path = config.path

if __name__ == "__main__":
    for frame_idx in range(frame_num):
        yuvs_dir = path + "/frames/frame_{}/yuvs".format(frame_idx)
        print(yuvs_dir)
        if os.path.exists(yuvs_dir):
            os.system(f"rm -r {yuvs_dir}")