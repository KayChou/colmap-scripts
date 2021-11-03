import config
import os

frame_num = 300
camera_num = config.camera_num

path = config.path + "/frames_afterprocess"

for frame_idx in range(frame_num):
    images_dir = path + "/frame_{}/images/".format(frame_idx)
    depths_dir = path + "/frame_{}/depth_bg2/".format(frame_idx)
    yuv_rgb = path + "/frame_{}/image_yuv/".format(frame_idx)
    yuv_depth = path + "/frame_{}/depth_yuv/".format(frame_idx)

    os.makedirs(yuv_rgb, exist_ok=True)
    os.makedirs(yuv_depth, exist_ok=True)

    print("processing ", images_dir)

    for cam_idx in range(camera_num):
        # images
        filename = f"{images_dir}{cam_idx}.png"
        output = f"{yuv_rgb}{cam_idx}.yuv"
        cmd = "ffmpeg -s 1920x1080 -pix_fmt rgb24 -i {} {} -y -loglevel quiet".format(filename, output)
        os.system(cmd)

        # depth
        filename = f"{depths_dir}{cam_idx}.png"
        output = f"{yuv_depth}{cam_idx}.yuv"
        cmd = "ffmpeg -s 1920x1080 -pix_fmt rgb24 -i {} {} -y -loglevel quiet".format(filename, output)
        os.system(cmd)