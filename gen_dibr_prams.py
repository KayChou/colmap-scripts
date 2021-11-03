import config
from scipy.spatial.transform import Rotation as _R
import numpy as np

input_path = config.created_dir
camera_num = config.camera_num

path = config.path
v_cfg = input_path + "v10.cfg"
ref_cfg = input_path + "para_cameras.txt"

min_depth = 14.849335
max_depth = 27.859304


def gen_virtual_view_cfg():
    f = open(v_cfg, 'w')
    f.write("{:<18} = {}\n".format("CamNum", camera_num))
    f.write("{:<18} = {}\n".format("SourceWidth", 1920))
    f.write("{:<18} = {}\n".format("SourceHeight", 1080))
    f.write("{:<18} = {}\n".format("Color_Path", path + "/frames_afterprocess"))
    f.write("{:<18} = {}\n".format("Depth_Path", path + "/frames_afterprocess"))
    f.write("{:<18} = {}\n".format("Cam_Params_File", ref_cfg))
    f.write("{:<18} = {}\n".format("Color_Output_File", 0.00))
    f.write("{:<18} = {}\n".format("Depth_Output_File", 0.00))

    f.write("{:<18} = {}\n".format("Vcam_krt_R_0", 0.00))
    f.write("{:<18} = {}\n".format("Vcam_krt_R_1", 0.00))
    f.write("{:<18} = {}\n".format("Vcam_krt_R_2", 0.00))
    f.write("{:<18} = {}\n".format("Vcam_krt_R_3", 0.00))
    f.write("{:<18} = {}\n".format("Vcam_krt_R_4", 0.00))
    f.write("{:<18} = {}\n".format("Vcam_krt_R_5", 0.00))
    f.write("{:<18} = {}\n".format("Vcam_krt_R_6", 0.00))
    f.write("{:<18} = {}\n".format("Vcam_krt_R_7", 0.00))
    f.write("{:<18} = {}\n".format("Vcam_krt_R_8", 0.00))
    f.write("\n")

    f.write("{:<18} = {}\n".format("Vcam_krt_WorldPosition_0", 0.00))
    f.write("{:<18} = {}\n".format("Vcam_krt_WorldPosition_1", 0.00))
    f.write("{:<18} = {}\n".format("Vcam_krt_WorldPosition_2", 0.00))
    f.write("{:<18} = {}\n".format("Vcam_krt_kc_0", 0.00))
    f.write("{:<18} = {}\n".format("Vcam_krt_kc_1", 0.00))
    f.write("{:<18} = {}\n".format("Vcam_krt_kc_2", 0.00))
    f.write("{:<18} = {}\n".format("Vcam_krt_cc_0", 0.00))
    f.write("{:<18} = {}\n".format("Vcam_krt_cc_1", 0.00))
    f.write("{:<18} = {}\n".format("lens_fov", 0.00))
    f.write("{:<18} = {}\n".format("fisheye_radius", 0.00))
    f.write("{:<18} = {}\n".format("Vcam_src_width", 1920))
    f.write("{:<18} = {}\n".format("Vcam_src_height", 1080))
    f.write("{:<18} = {}\n".format("Vcam_ltype", 0.00))
    
    f.close()


def gen_extrinsics():
    f = open(ref_cfg, 'w')
    f_int = open(input_path + "cameras.txt", 'r')
    f_ext = open(input_path + "images.txt", 'r')
    f_vis = open(input_path + "visulization.txt", 'w')

    lines_int = f_int.readlines()
    lines_ext = f_ext.readlines()

    f.write("{} {}\n".format(min_depth, max_depth))

    for cam_idx in range(camera_num):
        intrinsics = lines_int[cam_idx].strip().split(" ")
        extrinsics = lines_ext[2 * cam_idx].strip().split(" ")

        R_quat_raw = extrinsics[1:5]
        R_quat = [R_quat_raw[1], R_quat_raw[2], R_quat_raw[3], R_quat_raw[0]]
        R = _R.from_quat(R_quat).as_matrix()

        t = np.array(extrinsics[5:8]).astype(np.float64)

        w0 = R[0][0] * t[0] + R[1][0] * t[1] + R[2][0] * t[2]
        w1 = R[0][1] * t[0] + R[1][1] * t[1] + R[2][1] * t[2]
        w2 = R[0][2] * t[0] + R[1][2] * t[1] + R[2][2] * t[2]

        world_position = [-w0, -w1, -w2]

        f.write(f"camera_id {cam_idx}\n")
        f.write(f"resolution {1920} {1080}\n")
        f.write(f"K_matrix {intrinsics[4]} {intrinsics[5]} {intrinsics[6]} {intrinsics[7]}\n")
        f.write(f"R_matrix {R[0][0]} {R[0][1]} {R[0][2]} {R[1][0]} {R[1][1]} {R[1][2]} {R[2][0]} {R[2][1]} {R[2][2]}\n")
        f.write(f"world_position {world_position[0]} {world_position[1]} {world_position[2]}\n")

        f_vis.write(f"{0} {R[0][0]} {R[0][1]} {R[0][2]} {R[1][0]} {R[1][1]} {R[1][2]} {R[2][0]} {R[2][1]} {R[2][2]} {world_position[0]} {world_position[1]} {world_position[2]} {0} 0.png\n")

    f.close()
    f_ext.close()
    f_int.close()
    f_vis.close()


def gen_colmap_rt():
    f_int = open(input_path + "cameras.txt", 'r')
    f_ext = open(input_path + "images.txt", 'r')

    f_colmap_rt = open(input_path + "colmap_rt.txt", 'w')

    lines_int = f_int.readlines()
    lines_ext = f_ext.readlines()

    for cam_idx in range(camera_num):
        intrinsics = lines_int[cam_idx].strip().split(" ")
        extrinsics = lines_ext[2 * cam_idx].strip().split(" ")

        R_quat_raw = extrinsics[1:5]
        R_quat = [R_quat_raw[1], R_quat_raw[2], R_quat_raw[3], R_quat_raw[0]]
        R = _R.from_quat(R_quat).as_matrix()

        t = np.array(extrinsics[5:8]).astype(np.float64)

        f_colmap_rt.write(f"{R[0][0]} {R[0][1]} {R[0][2]} {R[1][0]} {R[1][1]} {R[1][2]} {R[2][0]} {R[2][1]} {R[2][2]} {t[0]} {t[1]} {t[2]} {1920} {1080} {intrinsics[4]} {intrinsics[5]}\n")

    f_colmap_rt.close()
    f_ext.close()
    f_int.close()




if __name__ == "__main__":
    # gen_virtual_view_cfg()
    # gen_extrinsics()
    gen_colmap_rt()