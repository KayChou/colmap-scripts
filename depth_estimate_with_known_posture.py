import os
import config
from update_instrinsic import camTodatabase
import time

camera_num = config.camera_num
frame_num = config.frame_num
path = config.path
created_dir = config.created_dir

CLEAN_MID_FILES = 1

if __name__ == "__main__":
    for frame_idx in range(0, frame_num, 1):
    # for frame_idx in config.un_valid:
        images_dir = path + "/frames/frame_{}/images/".format(frame_idx)
        dense_dir = path + "/frames/frame_{}/dense/".format(frame_idx)
        triangulate_dir = path + "/frames/frame_{}/triangulated/".format(frame_idx)

        print("processing {} | ".format(images_dir), end="")
        T1 = time.time()

        # # estimate all 12 cameras' posture for each frame
        if os.path.exists(dense_dir):
            os.system('rm -rf ' + dense_dir)
        if os.path.exists(triangulate_dir):
            os.system('rm -rf ' + triangulate_dir)
        os.system('mkdir ' + dense_dir)
        os.system('mkdir ' + triangulate_dir)
        
        os.system("colmap feature_extractor --database_path database.db --image_path {} --SiftExtraction.gpu_index=0 > /dev/null".format(images_dir))
        # camTodatabase("database.db", "{}cameras.txt".format(created_dir))
        os.system(f"colmap exhaustive_matcher --database_path database.db > /dev/null")
        os.system(f"colmap point_triangulator --database_path database.db --image_path {images_dir} --input_path {created_dir} --output_path {triangulate_dir} > /dev/null")
        os.system(f"colmap model_converter --input_path {triangulate_dir} --output_path {triangulate_dir} --output_type TXT > /dev/null")
        os.system(f"colmap image_undistorter --image_path {images_dir} --input_path {triangulate_dir} --output_path {dense_dir} > /dev/null")
        os.system(f"colmap patch_match_stereo --workspace_path {dense_dir} --workspace_format COLMAP --PatchMatchStereo.geom_consistency true > /dev/null")
        # os.system(f"colmap stereo_fusion --workspace_path {dense_dir} --output_path {dense_dir}/fused.ply")
        os.system('rm database.db')

        if CLEAN_MID_FILES:
            os.system("rm -r {}".format(triangulate_dir))
        
        T2 = time.time()
        print('time cost: %f min' % (float(T2 - T1) / 60))

