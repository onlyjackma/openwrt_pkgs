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

#define FEATURE_FILE "./feature.db"
#define MODULE_PATH "../models"

using namespace std;

struct img_feature{
    string img_path;
    string name;
    std::vector<float> feature;
};

struct RecognitionResult{
    float ppoint[10];
	cv::Rect rect;
	std::string name;

};

std::vector<img_feature> features;
MTCNN *mtcnn ;
MobileFaceNet *mobilefacenet;
ofstream of;

void usage(char *proc)
{
   printf("usage: %s <path to feature images>\n",proc);
   exit(1);
}

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


int get_image_info(char *path)
{
    DIR *dir;
    struct dirent *ent;
    struct img_feature imf;
    string str;

    if((dir = opendir(path))!= NULL){
        while((ent = readdir(dir)) != NULL){
            if(ent->d_type == DT_REG){
                printf("type %d, name %s\n",ent->d_type,ent->d_name);
                imf.name = ent->d_name;
                imf.img_path=str.append(path).append("/").append(ent->d_name);
                str = "";
                features.push_back(imf);
            }
        }
        closedir(dir);
        return 0;
    }else{
        printf("Open image path error");
    }
   
}

std::vector<float> gen_feature(cv::Mat image, string path)
{
    //cv::Mat image;
    double dst_landmark[10] = {
                38.2946, 73.5318, 55.0252, 41.5493, 70.7299,
                51.6963, 51.5014, 71.7366, 92.3655, 92.2041 };
    int nface = 0;
    int num_box = 0;
    std::vector<Bbox> bboxes;
	std::vector<RecognitionResult> ret;
    std::vector<float> feature;

    //image = cv::imread(path,1);
    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB,image.cols, image.rows);
    cout << image.cols << "  " << image.rows <<endl;
    mtcnn->detect(ncnn_img,bboxes);
    printf("111\n");
    num_box = bboxes.size();
    cout << num_box << endl;
    if(num_box > 1 || num_box <= 0){
        printf("Too many faces or no face find int %s, please give proper images\n",path.c_str());
        return feature;
    }else{
        for(int i = 0 ; i < num_box ; i++){
            cv::Mat face;
            std::vector<cv::Point2f> coord5points;
	        std::vector<cv::Point2f> facePointsByMtcnn;
            
            cv::Rect face_rect(bboxes[i].x1 ,bboxes[i].y1 , bboxes[i].x2 - bboxes[i].x1, bboxes[i].y2 - bboxes[i].y1);
            //cout << bboxes[i].x1<<bboxes[i].y1<<bboxes[i].x2 << bboxes[i].y2 << endl;
            for (int j = 0; j < 5; j ++) {
	            facePointsByMtcnn.push_back(cvPoint(bboxes[i].ppoint[j], bboxes[i].ppoint[j + 5]));
	            coord5points.push_back(cv::Point2f(dst_landmark[j], dst_landmark[j + 5]));
	        }
            cv::Mat warp_mat = estimateRigidTransform(facePointsByMtcnn, coord5points, false);
            if(warp_mat.empty()){
                warp_mat = getsrc_roi2(facePointsByMtcnn,coord5points);
            }
            warp_mat.convertTo(warp_mat,CV_32FC1);
            face = cv::Mat::zeros(112,112,image.type());
            warpAffine(image, face, warp_mat, face.size());
            //feature.clear();
            mobilefacenet->start(face,feature);
            return feature;
        }
    }
    

}

int save_feature(string name , vector<float> feature)
{
    //ofstream of(FEATURE_FILE);
    vector<float>::iterator it;
    of << name << ",";
    //it = feature.begin();
    for (int i = 0; i != feature.size(); i++) {
        of << feature[i] << ",";
    }
    of << endl;

}

string get_name(string pname)
{

    int idx = pname.find_first_of('.');
    return pname.substr(0,idx);
    // char *rest = pname.c_str();
    //  strtok_r(pname.c_str(),sp);
}

void do_some_init()
{
    mtcnn = new MTCNN(MODULE_PATH);
    mobilefacenet = new MobileFaceNet(MODULE_PATH);
    of.open(FEATURE_FILE);
}

int main(int argc ,char *argv[])
{
    char *path;
    DIR *dir;
    cv::Mat image;
    struct dirent *ent;
    std::vector<string> img_paths;
    if(argc < 2) {
        usage(argv[0]);
    }
    path = argv[1];
    printf("The images file path is %s\n",path);
    do_some_init();
    get_image_info(path);

   for(std::vector<img_feature>::iterator it = features.begin();it != features.end();it++){
        image = cv::imread(it->img_path,1);
        it->feature = gen_feature(image, it->img_path);
        if(!it->feature.empty()){
            it->name = get_name(it->name);
            save_feature(it->name,it->feature);
            printf("name:%s path:%s\n",it->name.c_str(),it->img_path.c_str());
            cout << "feature_size :" << it->feature.size() << endl;
        }
        //cv::imshow("IMAGE SHOW",image);
        //cvWaitKey(0);
   }
   
     
    of.close();

    return 0;
}
