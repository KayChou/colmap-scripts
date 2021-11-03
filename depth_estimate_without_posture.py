import os
import config

camera_num = config.camera_num
frame_num = config.frame_num
path = config.path

if __name__ == "__main__":
    for frame_idx in range(frame_num):
        images_dir = path + "/frames/frame_{}/images/".format(frame_idx)
        sparse_dir = path + "/frames/frame_{}/sparse/".format(frame_idx)
        dense_dir = path + "/frames/frame_{}/dense/".format(frame_idx)

        # estimate all 12 cameras' posture for each frame
        if os.path.exists(sparse_dir):
            os.system('rm -r ' + sparse_dir)
        os.system('mkdir ' + sparse_dir)
        if os.path.exists(dense_dir):
            os.system('rm -r ' + dense_dir)
        os.system('mkdir ' + dense_dir)
        
        # os.system("colmap feature_extractor --database_path database.db --image_path {}".format(images_dir))
        os.system("colmap feature_extractor --database_path database.db --image_path {} --ImageReader.camera_model PINHOLE".format(images_dir))
        os.system("colmap exhaustive_matcher --database_path database.db")
        os.system("colmap mapper --database_path database.db --image_path {} --output_path {}".format(images_dir, sparse_dir))
        os.system("colmap image_undistorter --image_path {} --input_path {}0/ --output_path {}".format(images_dir, sparse_dir, dense_dir))
        # os.system("colmap patch_match_stereo --workspace_path dataset2/frame_{}/dense/ --workspace_format COLMAP --PatchMatchStereo.geom_consistency true".format(frame_idx))
        os.system('rm database.db')

        # convert cameras.bin to cameras.txt
        os.system("colmap model_converter --input_path {}0 --output_path {} --output_type TXT".format(sparse_dir, sparse_dir))
        os.system("colmap model_converter --input_path {}sparse --output_path {}sparse --output_type TXT".format(dense_dir, dense_dir))
        