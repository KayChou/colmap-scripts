import sys 
sys.path.append("..") 
import config
import os

camera_num = config.camera_num

if __name__ == "__main__":
    for cam_idx in range(camera_num):
        os.system(f"./build/main {cam_idx} &")