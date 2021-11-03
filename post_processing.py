import collections
import config
import cv2
import numpy as np

path = config.path
frame_num = 100 # config.frame_num
camera_num = config.camera_num
background_path = "/run/media/benjamin/HDD-3/Dataset/medialab_20210924/background"


# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)


def morphological_filter(depth_map):
    depth_map = cv2.dilate(depth_map, FULL_KERNEL_3, iterations=1)
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_3)
    return depth_map



if __name__ == "__main__":
    for cam_idx in range(camera_num):
        depth_img = cv2.imread(f"{background_path}/depths/{cam_idx}.png")
        depth_img = np.float32(depth_img)
        depth_img = np.mean(depth_img, axis = 2)

        depth_img = morphological_filter(depth_img)
        
        cv2.imwrite(f"{background_path}/depths/post_{cam_idx}.png", depth_img)