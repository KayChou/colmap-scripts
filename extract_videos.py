import os
import config

camera_num = config.camera_num
frame_num = config.frame_num
start_frame_idx = config.start_frame_idx
path = config.path

bias_frame = 15

if __name__ == "__main__":
    # # extract videos to images
    video_path = path + "/videos/"
    for cam_idx in range(camera_num):
        video_name = video_path + f"{cam_idx + 1}" + ".mp4"

        if not os.path.exists(video_path + "camera{}".format(cam_idx)):
            os.system('mkdir {}camera{}'.format(video_path, cam_idx))

        cmd = "ffmpeg -i {} -filter:v select=\"between(n\, {}\, {})\" -pix_fmt rgb24 {}camera{}/%d.png".format(video_name, start_frame_idx - bias_frame, start_frame_idx + frame_num - 1, video_path, cam_idx)
        print(cmd)
        os.system(cmd)
    
    if not os.path.exists(path + "/frames"):
        os.mkdir(path + "/frames")
    
    # mv images to frame_x
    for frame_idx in range(0, frame_num, 1):
        frame_path = path + "/frames/frame_{}".format(frame_idx)
        
        if not os.path.exists(frame_path):
            os.system('mkdir {}'.format(frame_path))
            os.system('mkdir {}/images'.format(frame_path))

        for cam_idx in range(camera_num):
            cmd = "cp {}camera{}/{}.png {}/images/{}.png".format(video_path, cam_idx, bias_frame + frame_idx, frame_path, cam_idx)
            os.system(cmd)
    
    for cam_idx in range(camera_num):
        os.system('rm -rf {}camera{}'.format(video_path, cam_idx))

