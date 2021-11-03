from PIL import Image
import numpy as np
import config
import math
import os
import cv2

from skimage.measure import compare_ssim

path = config.path
frame_num = config.frame_num
camera_num = config.camera_num

background_path = "/run/media/benjamin/HDD-3/Dataset/medialab_20210924/background"


def read_all_depths_of_one_cam(cam_idx):
    depth_images = np.zeros([frame_num, 1080, 1920])
    valid_frame = 0

    for frame_idx in range(frame_num):
        depth_path = path + f"/frames/frame_{frame_idx}/depths/bg_{cam_idx}.png"

        if not os.path.exists(depth_path):
            continue

        depth_img = np.array(Image.open(depth_path)) # [1080, 1920]
        depth_img = np.mean(depth_img, axis = 2)
        depth_images[valid_frame, :, :] = depth_img[:, :]
        valid_frame = valid_frame + 1

    print(f"current cam: {cam_idx} has {valid_frame} valid frames")
    depth_images = depth_images[0:valid_frame, :, :]
    # depth_out = np.mean(depth_images, axis = 0)
    depth_out = np.zeros([1080, 1920])
    for h in range(1080):
        for w in range(1920):
            list = depth_images[:, h, w]
            
            if(np.count_nonzero(list) > 5):
                if(np.max(list) - np.min(list[np.nonzero(list)])) > 20:
                    list[list > (np.min(list[np.nonzero(list)]) + 20)] = 0

                depth_out[h, w] = np.sum(list) / np.count_nonzero(list)

    depth_out = Image.fromarray(np.uint8(depth_out))
    depth_out.save(f"{background_path}/depths/{cam_idx}.png")
    


def cut_two_images(image_l, image_r):
    spice = 1100
    img_out = np.zeros([1080, 1920, 3])
    img_l = np.array(Image.open(image_l)) # 1080, 1920
    img_r = np.array(Image.open(image_r))

    img_out[:, 0:spice, :] = img_l[:, 0:spice, :]
    img_out[:, spice:1920, :] = img_r[:, spice:1920, :]

    img_out = Image.fromarray(np.uint8(img_out))
    img_out.save("12.png")


if __name__ == "__main__":
    
    for cam_idx in range(camera_num):
        read_all_depths_of_one_cam(cam_idx)
    # image_l = "/run/media/benjamin/HDD-3/Dataset/medialab_20210924/basketball/videos/images/2.png"
    # image_r = "/run/media/benjamin/HDD-3/Dataset/medialab_20210924/basketball/videos/images/130.png"
    # cut_two_images(image_l, image_r)
    