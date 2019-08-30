#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h>
#include <string.h>
#include <arpa/inet.h>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <iostream>
#include <dirent.h>
#include "mobilefacenet.h"
#include "mtcnn.h"
#include "featuredb.h"
using namespace std;
using namespace cv;

struct RecognitionResult{
    float ppoint[10];
	cv::Rect rect;
	std::string name;

};

#define MODULE_PATH "/lib/face/models"
char *url;
MTCNN *mtcnn;
MobileFaceNet *mobilefacenet;
FeatureDB *featuredb;
RecognitionResult result;
std::vector<cv::Mat> frames;
pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;

cv::Mat getsrc_roi2(std::vector<cv::Point2f> x0, std::vector<cv::Point2f> dst)
{
    int size = dst.size();
    cv::Mat A = cv::Mat::zeros(size * 2, 4, CV_32FC1);
    cv::Mat B = cv::Mat::zeros(size * 2, 1, CV_32FC1);

    for (int i = 0; i < size; i++) {
        A.at<float>(i << 1, 0) = x0[i].x;// roi_dst[i].x;
        A.at<float>(i << 1, 1) = -x0[i].y;
        A.at<float>(i << 1, 2) = 1;
        A.at<float>(i << 1, 3) = 0;
        A.at<float>(i << 1 | 1, 0) = x0[i].y;
        A.at<float>(i << 1 | 1, 1) = x0[i].x;
        A.at<float>(i << 1 | 1, 2) = 0;
        A.at<float>(i << 1 | 1, 3) = 1;

        B.at<float>(i << 1) = dst[i].x;
        B.at<float>(i << 1 | 1) = dst[i].y;
    }

    cv::Mat roi = cv::Mat::zeros(2, 3, A.type());
    cv::Mat AT = A.t();
    cv::Mat ATA = A.t() * A;
    cv::Mat R = ATA.inv() * AT * B;

    roi.at<float>(0, 0) = R.at<float>(0, 0);
    roi.at<float>(0, 1) = -R.at<float>(1, 0);
    roi.at<float>(0, 2) = R.at<float>(2, 0);
    roi.at<float>(1, 0) = R.at<float>(1, 0);
    roi.at<float>(1, 1) = R.at<float>(0, 0);
    roi.at<float>(1, 2) = R.at<float>(3, 0);
    return roi;
}

void *do_face_recognization(void *para)
{
    vector<Mat>::iterator fit;
    cv::Mat image;
    ncnn::Mat ncnn_img;
    std::vector<Bbox> bboxes;
    int num_box;
    double dst_landmark[10] = {
                38.2946, 73.5318, 55.0252, 41.5493, 70.7299,
                51.6963, 51.5014, 71.7366, 92.3655, 92.2041 };
    int nface=0;
    cout << "start face rec\n" << endl;
    while(1){
        usleep(10);
        if(!frames.empty()){
            pthread_mutex_lock(&mutex1);
            fit = frames.begin();
            image = (*fit).clone();
            frames.erase(fit);
            pthread_mutex_unlock(&mutex1);
            ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
            mtcnn->detect(ncnn_img,bboxes);
			//ncnn_img.release();
            num_box = bboxes.size();
            for(int i=0; i< num_box;i++){
                cv::Mat face;
                vector<cv::Point2f> coord5points;
	            vector<cv::Point2f> facePointsByMtcnn;
                cv::Rect rect(bboxes[i].x1, bboxes[i].y1, bboxes[i].x2 - bboxes[i].x1 , bboxes[i].y2 - bboxes[i].y1);
                for (int j = 0; j < 5; j ++) {
                    facePointsByMtcnn.push_back(cvPoint(bboxes[i].ppoint[j], bboxes[i].ppoint[j + 5]));
                    coord5points.push_back(cv::Point2f(dst_landmark[j], dst_landmark[j + 5]));
                    //cout << j;
	            }
            cv::Mat warp_mat = estimateRigidTransform(facePointsByMtcnn, coord5points, false);
	        if (warp_mat.empty()) {
	            warp_mat = getsrc_roi2(facePointsByMtcnn, coord5points);
	        }
	        warp_mat.convertTo(warp_mat, CV_32FC1);
	        face = cv::Mat::zeros(112, 112, image.type());
	        warpAffine(image, face, warp_mat, face.size());
            std::vector<float> feature;
	        mobilefacenet->start(face, feature);
            std::string name = featuredb->find_name(feature);
            result.name = name;
            result.rect = rect;
            std::cout << "name:" << result.name << endl;
			//image.release();
            }
        }
    }
}

void *do_img_capture(void *para)
{
    int fps;
    int ret;
    unsigned long frame_idx = 0;
    cv::Mat frame;

RE_OPEN:
    cv::VideoCapture cap(url);
    fps = cap.get(CAP_PROP_FPS);
    cout << fps << endl;

    if(!cap.isOpened()){
        cout << "open camera failed\n" << endl;
    }
    while(1){
        ret = cap.read(frame);
        if(!ret){
            cout << "read frame error\n" << endl;
            cap.release();
            goto RE_OPEN;
        }
        frame_idx = cap.get(cv::CAP_PROP_POS_FRAMES);
        if((frame_idx%fps) == 0){
            pthread_mutex_lock(&mutex1);
            frames.push_back(frame);
			//frame.release();
            pthread_mutex_unlock(&mutex1);
        }
    }
    cap.release();


}

int do_face_rec_init()
{
    mtcnn = new MTCNN(MODULE_PATH);
    mobilefacenet = new MobileFaceNet(MODULE_PATH);
    featuredb = new FeatureDB(MODULE_PATH, 0.70);
}

int main(int argc, char *argv[])
{
    pthread_t cap_thread,rec_thread;
    int  ret;

    if(argc < 2 ){
        printf("%s <camera url>",argv[0]);
        exit(1);
    }

    url = strdup(argv[1]);
    if(url == NULL){
        printf("invalid camera url\n");
        exit(1);
    }
    do_face_rec_init();

    ret = pthread_create(&cap_thread,NULL , do_img_capture,NULL);
    if(ret < 0){
        printf("pthread create cap thread failed\n");
        return -1;
    }

    ret = pthread_create(&rec_thread ,NULL , do_face_recognization,NULL);
    if(ret < 0){
        printf("pthread create face recognize thread failed\n");
        return -1;
    }

    pthread_join(cap_thread,NULL);
    pthread_join(rec_thread,NULL);


}
