#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "opencv/cv.h"
#include "string"
#include <thread>
#include <stdlib.h>

#define camera_num 12
#define frame_num 250
#define root_path "/run/media/benjamin/HDD-3/Dataset/medialab_20210924/basketball"
#define background_path "/run/media/benjamin/HDD-3/Dataset/medialab_20210924/background"
#define patch_size 20
#define gaussian_size 5


using namespace std;
using namespace cv;


// compute the ssim of two patch
float compute_ssim(cv::Mat &i1, cv::Mat & i2) {
    const float C1 = 6.5025, C2 = 58.5225;
    int d = CV_32F;
    cv::Mat I1, I2;
    i1.convertTo(I1, d);
    i2.convertTo(I2, d);
    cv::Mat I1_2 = I1.mul(I1);
    cv::Mat I2_2 = I2.mul(I2);
    cv::Mat I1_I2 = I1.mul(I2);
    cv::Mat mu1, mu2;
    GaussianBlur(I1, mu1, cv::Size(gaussian_size, gaussian_size), 1.5);
    GaussianBlur(I2, mu2, cv::Size(gaussian_size, gaussian_size), 1.5);
    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);
    cv::Mat sigma1_2, sigam2_2, sigam12;
    GaussianBlur(I1_2, sigma1_2, cv::Size(gaussian_size, gaussian_size), 1.5);
    sigma1_2 -= mu1_2;
 
    GaussianBlur(I2_2, sigam2_2, cv::Size(gaussian_size, gaussian_size), 1.5);
    sigam2_2 -= mu2_2;
 
    GaussianBlur(I1_I2, sigam12, cv::Size(gaussian_size, gaussian_size), 1.5);
    sigam12 -= mu1_mu2;
    cv::Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigam12 + C2;
    t3 = t1.mul(t2);
 
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigam2_2 + C2;
    t1 = t1.mul(t2);
 
    cv::Mat ssim_map;
    divide(t3, t1, ssim_map);
    cv::Scalar mssim = mean(ssim_map);
 
    float ssim = (mssim.val[0] + mssim.val[1] + mssim.val[2]) /3;
    return ssim;
}


// compute the similarity(ncc) of two images with the center in the (h, w)
float compute_patch_similarity(int h, int w, cv::Mat img1, cv::Mat img2) {
    int start_h, end_h;
    int start_w, end_w;

    if(h < 3) {
        start_h = int(max(0, h - patch_size / 2));
        end_h = int(min(1080 - 1, start_h + patch_size));
    }
    else {
        end_h = int(min(1080 - 1, h + patch_size / 2));
        start_h = int(max(0, end_h - patch_size));
    }
        
    if(w < 3) {
        start_w = int(max(0, w - patch_size / 2));
        end_w = int(min(1920 - 1, start_w + patch_size));
    }
    else {
        end_w = int(min(1920 - 1, w + patch_size / 2));
        start_w = int(max(0, end_w - patch_size));
    }
    cv::Rect rect(start_w, start_h, patch_size, patch_size);

    cv::Mat patch1 = img1(rect);
    cv::Mat patch2 = img2(rect);

    float ssim = compute_ssim(patch1, patch2);
    return ssim;      
}

// for one camera, read all frames of current camera, and split background
void read_all_depths_of_one_cam(int cam_idx) {
    char color_path[512];
    char color_back_path[512];
    char depth_path[512];
    char depth_out_path[512];

    cv::Mat color_img_back;
    cv::Mat color_img;

    for(int frame_idx = 120; frame_idx < frame_num; frame_idx++) {
        cv::Mat depth_img;

        sprintf(color_path, "%s/frames/frame_%d/images/%d.png", root_path, frame_idx, cam_idx);
        sprintf(depth_path, "%s/frames/frame_%d/depths/%d.png", root_path, frame_idx, cam_idx);
        sprintf(depth_out_path, "%s/frames/frame_%d/depths/bg_%d.png", root_path, frame_idx, cam_idx);
        sprintf(color_back_path, "%s/images/%d.png", background_path, cam_idx + 1);

        printf("%s\n", depth_path);

        color_img = cv::imread(color_path);
        depth_img = cv::imread(depth_path);
        color_img_back = cv::imread(color_back_path);

        int width = color_img.size[1];
        int height = color_img.size[0];

        float similarity = 0;

        // loop all images pixels, if ssim > 0.9, then current pixel is valid
        for(int h = 0; h < height; h++) {
            for(int w = 0; w < width; w++) {
                similarity = compute_patch_similarity(h, w, color_img, color_img_back);

                if(similarity < 0.85) {
                    depth_img.at<cv::Vec3b>(h, w)[0] = 0; // (ncc + 0) * 255 / 1;
                    depth_img.at<cv::Vec3b>(h, w)[1] = 0; // (ncc + 0) * 255 / 1;
                    depth_img.at<cv::Vec3b>(h, w)[2] = 0; // (ncc + 0) * 255 / 1;
                }
            }
        }
        cv::imwrite(depth_out_path, depth_img);
    }
}


int main(int argc, char *argv[])
{
    int cam_idx = atoi(argv[1]);
    read_all_depths_of_one_cam(cam_idx);
}