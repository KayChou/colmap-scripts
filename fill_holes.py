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

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)


def fill_in_multiscale(depth_map,
                       dilation_kernel_far=CROSS_KERNEL_7,
                       dilation_kernel_med=CROSS_KERNEL_5,
                       dilation_kernel_near=CROSS_KERNEL_3,
                       extrapolate=False,
                       blur_type='bilateral'):
    """Slower, multi-scale dilation version with additional noise removal that
    provides better qualitative results.

    Args:
        depth_map: projected depths
        dilation_kernel_far: dilation kernel to use for 30.0 < depths < 80.0 m
        dilation_kernel_med: dilation kernel to use for 15.0 < depths < 30.0 m
        dilation_kernel_near: dilation kernel to use for 0.1 < depths < 15.0 m
        extrapolate:whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'gaussian' - provides lower RMSE
            'bilateral' - preserves local structure (recommended)
        show_process: saves process images into an OrderedDict

    Returns:
        depth_map: dense depth map
        process_dict: OrderedDict of process images
    """

    # Convert to float32
    depths_in = np.float32(depth_map)

    # Calculate bin masks before inversion
    valid_pixels_far = (depths_in > 0.1) & (depths_in <= 50.0)
    valid_pixels_med = (depths_in > 50.0) & (depths_in <= 200.0)
    valid_pixels_near = (depths_in > 200.0)

    # Invert (and offset)
    s1_inverted_depths = np.copy(depths_in)
    valid_pixels = (s1_inverted_depths > 0.1)

    # Multi-scale dilation
    dilated_far = cv2.dilate(np.multiply(s1_inverted_depths, valid_pixels_far), dilation_kernel_near)
    dilated_med = cv2.dilate(np.multiply(s1_inverted_depths, valid_pixels_med), dilation_kernel_near)
    dilated_near = cv2.dilate(np.multiply(s1_inverted_depths, valid_pixels_near), dilation_kernel_near)

    # Find valid pixels for each binned dilation
    valid_pixels_near = (dilated_near > 0.1)
    valid_pixels_med = (dilated_med > 0.1)
    valid_pixels_far = (dilated_far > 0.1)

    # Combine dilated versions, starting farthest to nearest
    s2_dilated_depths = np.copy(s1_inverted_depths)
    s2_dilated_depths[valid_pixels_far] = dilated_far[valid_pixels_far]
    s2_dilated_depths[valid_pixels_med] = dilated_med[valid_pixels_med]
    s2_dilated_depths[valid_pixels_near] = dilated_near[valid_pixels_near]

    # Small hole closure
    s3_closed_depths = cv2.morphologyEx(s2_dilated_depths, cv2.MORPH_CLOSE, FULL_KERNEL_3)

    # Median blur to remove outliers
    s4_blurred_depths = np.copy(s3_closed_depths)
    blurred = cv2.medianBlur(s3_closed_depths, 5)
    valid_pixels = (s3_closed_depths > 0.1)
    s4_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Calculate a top mask
    top_mask = np.ones(depths_in.shape, dtype=np.bool)
    for pixel_col_idx in range(s4_blurred_depths.shape[1]):
        pixel_col = s4_blurred_depths[:, pixel_col_idx]
        top_pixel_row = np.argmax(pixel_col > 0.1)
        top_mask[0:top_pixel_row, pixel_col_idx] = False

    # Get empty mask
    valid_pixels = (s4_blurred_depths > 0.1)
    empty_pixels = ~valid_pixels & top_mask

    # Hole fill
    dilated = cv2.dilate(s4_blurred_depths, FULL_KERNEL_5)
    s5_dilated_depths = np.copy(s4_blurred_depths)
    s5_dilated_depths[empty_pixels] = dilated[empty_pixels]

    # # Extend highest pixel to top of image or create top mask
    # s6_extended_depths = np.copy(s5_dilated_depths)
    # top_mask = np.ones(s5_dilated_depths.shape, dtype=np.bool)

    # top_row_pixels = np.argmax(s5_dilated_depths > 0.1, axis=0)
    # top_pixel_values = s5_dilated_depths[top_row_pixels,
    #                                      range(s5_dilated_depths.shape[1])]

    # for pixel_col_idx in range(s5_dilated_depths.shape[1]):
    #     if extrapolate:
    #         s6_extended_depths[0:top_row_pixels[pixel_col_idx],
    #                            pixel_col_idx] = top_pixel_values[pixel_col_idx]
    #     else:
    #         # Create top mask
    #         top_mask[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = False

    # # Fill large holes with masked dilations
    # s7_blurred_depths = np.copy(s6_extended_depths)
    # for i in range(6):
    #     empty_pixels = (s7_blurred_depths < 0.1) & top_mask
    #     dilated = cv2.dilate(s7_blurred_depths, FULL_KERNEL_5)
    #     s7_blurred_depths[empty_pixels] = dilated[empty_pixels]

    # # Median blur
    # blurred = cv2.medianBlur(s7_blurred_depths, 5)
    # valid_pixels = (s7_blurred_depths > 0.1) & top_mask
    # s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # if blur_type == 'gaussian':
    #     # Gaussian blur
    #     blurred = cv2.GaussianBlur(s7_blurred_depths, (5, 5), 0)
    #     valid_pixels = (s7_blurred_depths > 0.1) & top_mask
    #     s7_blurred_depths[valid_pixels] = blurred[valid_pixels]
    # elif blur_type == 'bilateral':
    #     # Bilateral blur
    #     blurred = cv2.bilateralFilter(s7_blurred_depths, 5, 0.5, 2.0)
    #     s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # # Invert (and offset)
    # s8_inverted_depths = np.copy(s7_blurred_depths)
    # valid_pixels = np.where(s8_inverted_depths > 0.1)

    return s5_dilated_depths


if __name__ == "__main__":
    for cam_idx in range(camera_num):
        depth_img = cv2.imread(f"{background_path}/depths/{cam_idx}.png")
        depth_img = np.float32(depth_img)
        depth_img = np.mean(depth_img, axis = 2)
        depth_out = fill_in_multiscale(depth_img, extrapolate=False, blur_type='bilateral')
        cv2.imwrite(f"{background_path}/depths/post_{cam_idx}.png", depth_out)